from hesp.util import segmenter_helpers
from hesp.config.config import Config
from hesp.embedding_space.hyperbolic_embedding_space import HyperbolicEmbeddingSpace
from hesp.embedding_space.euclidean_embedding_space import EuclideanEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util import loss
from hesp.models.DeepLabV3Plus_Pytorch import network
import torchmetrics
import random
import os
import torch
from hesp.util.norm_registry import NormRegistry
# from hesp.util.data_helpers import imshow

ROOT = '/home/mgonggri/master_thesis/'
NORM_TERM = True


class Segmenter(torch.nn.Module):
    def __init__(
        self,
        tree: Tree,
        config: Config,
        device,
        save_folder: str = "saves/",
        seed: float = None,
    ) -> None:        
        
        super().__init__()
        
        self.save_folder = save_folder
        self.config = config
        self.tree = tree
        self.seed = seed
        self.train_metrics = config.segmenter._TRAIN_METRICS    
        self.val_metrics = config.segmenter._VAL_METRICS
        self.device = device
        self.save_state = config.segmenter._SAVE_STATE
        self.weights = None
        
        f = self.config.segmenter._SAVE_FOLDER
        if self.save_state and not os.path.exists(f):
            os.mkdir(f)
        
        if self.config.dataset._NAME == 'pascal':
            i2c_file = config._ROOT + "datasets/pascal/PASCAL_i2c.txt"
            with open(i2c_file, "r") as f:
                self.i2c = {i : line.split(":")[1][:-1] for i, line in enumerate(f.readlines()) }
            
        if config.embedding_space._GEOMETRY == 'hyperbolic':
            self.embedding_space = HyperbolicEmbeddingSpace(tree, config)
        elif config.embedding_space._GEOMETRY == 'euclidean':
            self.embedding_space = EuclideanEmbeddingSpace(tree, config)
            
        self.embedding_space.return_projections = True
            
        self.embedding_model = network.modeling._load_model(
            arch_type=config.base_model_name,
            backbone=config.segmenter._BACKBONE,
            num_classes=config.segmenter._EFN_OUT_DIM,
            output_stride=config.segmenter._OUTPUT_STRIDE,
            pretrained_backbone=config.segmenter._PRE_TRAINED_BB
        )
        
        self.iou_fn = torchmetrics.classification.MulticlassJaccardIndex(
            config.dataset._NUM_CLASSES,
            average=None,
            ignore_index=255,
            validate_args=False
        )
        
        self.acc_fn = torchmetrics.classification.MulticlassAccuracy(
            config.dataset._NUM_CLASSES,
            average=None,
            multidim_average='global',
            ignore_index=255,
            validate_args=False
        )
        
        
    def MSNN_old(self, embeddings: torch.Tensor) -> torch.Tensor:
        """ 
        Normalize a batch of embeddings of shape (batch, dim, height, width) by the maximum norm over dim
        for every batch element and rescale them s.t. the maximum norm is equal to the radius 
        of the embedding space curvature: 1 / sqrt(c).
        """
        with torch.no_grad():
            norms = torch.linalg.vector_norm(embeddings, dim=1) 
            
            max_sample_norms = norms.amax(dim=(1, 2)) 
            
        normalized = embeddings / max_sample_norms[:, None, None, None] 
        
        rescaled_normalized = normalized * (1.0 / torch.sqrt(self.embedding_space.curvature))
        
        return rescaled_normalized
    
    
    # better version --> doesn't compute norms for masked pixels 
    def MSNN_new(self, embeddings) -> torch.Tensor:
        """ 
        Normalize a batch of embeddings of shape (batch, dim, height, width) by the maximum norm over dim
        for every batch element and rescale them s.t. the maximum norm is equal to the radius 
        of the embedding space curvature: 1 / sqrt(c). Take into account the pixel masking.
        """
        with torch.no_grad():
            # compute the norms of the pixels that are not masked out
            norms = torch.linalg.vector_norm(embeddings.moveaxis(1,-1)[self.valid_mask], dim=-1)
            
            # compute the slices that correspond to samples after masking
            slices = self.valid_mask.sum(dim=(1,2))

            # every sample has one maximum norm value
            sample_maxes = torch.ones(embeddings.size(0), 1, 1, 1,
                                       device=self.device)

            start = 0
            for i, s in enumerate(slices):
                end = start + s
                
                # catch case where the sample contains no valid embeddings;
                if not start == end:
                    # select the maximum norm for the unmasked embeddings in current sample
                    sample_maxes[i] = torch.amax(norms[start : end], dim=0)
                
                start = end

        # normalize the values in each batch element by their respective max norm
        # and rescale them by 1 / sqrt(c)
        return embeddings / sample_maxes * (1.0 / torch.sqrt(self.embedding_space.curvature))
            
    
    def scale_and_shift(self, embeddings: torch.Tensor) -> torch.Tensor:
            scale = 1.0 / 15.0
            offset = 0.0
            return embeddings * scale + offset
            
            
    def forward(self):
        """Call embedding model, normalize embeddings and compute class probs from these."""
        
        # make the pixel embeddings with the backbone model;
        embs = self.embedding_model(self.images) 
        
        # normalize the embeddings by the maximum norm in the same sample and rescale 
        # normalized_embs = self.MSNN_new(embs) 
        normalized_embs = self.scale_and_shift(embs)
        
        # perform multinomial logistic regression in the embedding space to compute the probs;
        self.proj_embs, self.cprobs = self.embedding_space(normalized_embs)
        
        # compute the norm weights 
        if self.config.segmenter._WEIGHTS != 'none':
            
            weight_args = {"embeddings"    : self.proj_embs,
                           "valid_mask"    : self.valid_mask,
                           "valid_labels"  : self.valid_labels,
                           "method"        : self.config.segmenter._WEIGHTS}
            
            self.weights = loss.compute_weights(**weight_args)
        
        # update the norm registry
        if self.config.segmenter._REGISTER_NORMS:
            
            self.norm_registry.update(
                self.valid_mask,
                self.valid_labels,
                self.proj_embs,
                self.batch_indices)
            
            if self.global_steps % self.config.segmenter._COLLECT_EVERY == 0:
                norm_avgs_per_class = self.norm_registry.average_per_class()
                print(
                    f'[mean norms per class]\n{norm_avgs_per_class}'
                )
                self.norm_storage.append(norm_avgs_per_class[None, :])
            
            
    def end_of_epoch(self):
        
        self.embedding_space.running_norms = 0.0
        self.steps = 0
        self.running_loss = 0.0
        
        if self.config.segmenter._REGISTER_NORMS:    
            self.norm_registry.average()
            print("[average norms per class]")
            print( self.norm_registry.average_per_class() )
        
        if self.train_metrics:
            self.compute_metrics('train')
            
        if self.val_metrics:
            self.compute_metrics_dataset(self.val_loader, 'val')
        
        self.steps = 0
        
        # if training stochastically, update the sample probs at the end of each epoch
        if self.config.segmenter._TRAIN_STOCHASTIC:
            self.compute_sample_probs()
        
        folder = self.config.segmenter._SAVE_FOLDER
        
        if self.acc_storage:   
            stored_accuracies = torch.cat(self.acc_storage, dim=0)
            stored_mious = torch.cat(self.miou_storage, dim=0)
            torch.save(stored_accuracies, f'{folder}accurarcies.pt')
            torch.save(stored_mious, f'{folder}mious.pt')
        
        if self.norm_storage: 
            stored_norms = torch.cat(self.norm_storage, dim=0)
            torch.save(stored_norms, f'{folder}norms.pt')

        
    def update_model(self):
        """ Compute the loss based on the probs and labels. Clip grads, loss backwards and optimizer step. """
        
        valid_cprobs = self.cprobs.moveaxis(1, -1)[self.valid_mask]
        
        embs, mask = None, None
        if NORM_TERM:
            embs = self.proj_embs
            mask = self.valid_mask
        
        hce_loss = loss.CCE(
            valid_cprobs,
            self.valid_labels,
            self.tree,
            self.weights,
            embs,
            mask)
        
        self.running_loss += hce_loss.item()
        
        self.print_intermediate(self.config.segmenter._COLLECT_EVERY)
        
        torch.nn.utils.clip_grad_norm_(
            self.embedding_space.offsets,
            self.config.segmenter._GRAD_CLIP)
        
        hce_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.steps += 1
        self.global_steps += 1
       
     
    def train_fn(self, train_loader, val_loader, optimizer, scheduler):
        self.init_training_states(train_loader, val_loader, optimizer, scheduler)
        
        self.acc_storage = []
        self.miou_storage = []
        self.norm_storage = []

        for edx in range(self.start_edx, self.config.segmenter._NUM_EPOCHS + self.start_edx, 1):
            self.edx = edx
            
            for images, labels, batch_indices in train_loader:
                self.labels = labels.to(self.device).squeeze()
                self.images = images.to(self.device)
                self.batch_indices = batch_indices
                self.valid_mask = self.labels <= self.tree.M - 1
                self.valid_labels = self.labels[self.valid_mask]
                
                # compute the class probabilities for each pixel
                self.forward()
                
                # compute the loss and update model parameters
                self.update_model()
                
                 # when using cyclic learning rate scheduling
                self.scheduler.step()  
                
                # print intermediate metrics
                if self.train_metrics:
                    self.metrics_step()
                    
                if self.global_steps % self.config.segmenter._COLLECT_EVERY == 0:
                    mious = self.iou_fn.compute().cpu()
                    accuracies = self.acc_fn.compute().cpu()
                    self.miou_storage.append(mious[None, :])
                    self.acc_storage.append(accuracies[None, :])
                
            # reset steps, increment epoch, take a scheduler step, and update parameters
            self.end_of_epoch()
            
        if self.save_state:
            self.save_states()
            print('Training done. Saving final model state..')
    
            
    def save_states(self) -> None:
        save_folder = self.config.segmenter._SAVE_FOLDER
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        torch.save(self.embedding_model.state_dict(), save_folder + 'embedding_model.pt')
        torch.save(self.embedding_space.state_dict(), save_folder + 'embedding_space.pt')
        torch.save(self.scheduler.state_dict(), save_folder + 'scheduler.pt')
        torch.save(self.optimizer.state_dict(), save_folder + 'optimizer.pt')
        
        with open(save_folder + 'epoch.txt', 'w') as f:
            f.write("{}\n{}".format(str(self.edx + 1), str(self.global_steps)))

            
    def collate_fn(self) -> None:
        """
        Given a dataset and a list of indices return a batch of samples.
        Returns a tuple of (batch_samples, batch_labels)
        """
        image_batch = []
        label_batch = []
        
        for idx in self.batch_indices:
            samples, labels, _ = self.dataset[idx]
            image_batch.append(samples[None, :])
            label_batch.append(labels)
            
        self.images = torch.cat(image_batch).to(self.device)
        self.labels = torch.cat(label_batch).to(self.device).squeeze()
        self.valid_mask = self.labels <= self.tree.M - 1
        self.valid_labels = self.labels[self.valid_mask]
        
        
    def init_training_states(
        self,
        train: torch.utils.data.Dataset  | torch.utils.data.DataLoader,
        val_loader : torch.utils.data.DataLoader,
        optimizer : torch.optim.Optimizer,
        scheduler : torch.optim.lr_scheduler,
        ) -> None:
        
        """ Initialize all the variables into self which are used for training given the configuration and setting. """
        EPS = 1e-6
        
        if self.config.segmenter._TRAIN_STOCHASTIC:
            print('Initializing stochastic training setup...')
            self.dataset = train
            self.num_samples = len(self.dataset)
            self.batch_size = self.config.segmenter._BATCH_SIZE

            # initialise all sample probabilities to be equally likely
            self.sample_probs = torch.ones(self.num_samples) / self.num_samples

            # initialise norms as a small number
            self.mean_projection_norms = torch.zeros(len(self.dataset), device=self.device) + EPS
            
            # projected embeddings are used to compute the sample probabilities
            self.embedding_space.return_projections = True

        else:
            self.num_samples = len(train.dataset)
            
        if self.config.segmenter._REGISTER_NORMS:
            self.norm_registry = NormRegistry(self.num_samples, self.config.dataset._NUM_CLASSES)
            
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.class_weights = torch.load(ROOT + "datasets/pascal/class_weights.pt").to(self.device)
        self.running_loss = 0.0
        self.global_steps = 0
        self.steps = 0
        
        if isinstance(self.seed, float):
            print('[random seed]  ', self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            
        self.init_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        
        if self.config.segmenter._RESUME:
            with open(self.save_folder + 'epoch.txt', 'r') as f:
                lines  = f.readlines()
                self.start_edx = int(lines[0])
                self.global_steps = int(lines[1])
                
            self.scheduler.load_state_dict(torch.load(self.save_folder + 'scheduler.pt'))
            self.optimizer.load_state_dict(torch.load(self.save_folder + 'optimizer.pt'))
        else:
            self.start_edx = 0 
    
    
    def compute_sample_probs(self) -> None:
        """
        Compute the probabilities of the samples by the inverse of their average norms
        such that larger norms make a sample less likely.
        """
        # compute the average norm per sample from the registry
        mean_sample_norms = self.norm_registry.average_per_sample()
        
        print(f"[sample norm range]: \
              \nmax  {mean_sample_norms.max().item()} \
              \nmean {mean_sample_norms.mean().item()} \
              \nmin  {mean_sample_norms.min().item()}")
        
        # replace zero norms by the average; can happen in case samples are unobserved / fully masked by ignore;
        mean_sample_norms[mean_sample_norms == 0.0] = mean_sample_norms.mean()
        
        # update the sample probabilities s.t. lower norms -> more likely to draw
        self.sample_probs = torch.softmax(1.0 / mean_sample_norms, dim=0)
        print(f"[sample probability range]: \
              max {self.sample_probs.max().item()}, \
              mean {self.sample_probs.mean().item()}, \
              min {self.sample_probs.min().item()}")
        
    
    def train_fn_stochastic(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs,
        ) -> None:
        
        self.acc_storage = []
        self.miou_storage = []
        self.norm_storage = []
            
        """ Probabilistic training loop that draws sample indices from a multinomial distribtution. """
        self.init_training_states(train_dataset, val_loader, optimizer, scheduler)
        torch.set_printoptions(sci_mode=False)
        
        # used for computing the sample probabilities
        
        #  for first epoch make sure each sample is seen at least once s.t. they all get norms; 
        indices = torch.multinomial(self.sample_probs, self.num_samples, replacement=False)
        
        for edx in range(self.start_edx, self.config.segmenter._NUM_EPOCHS, 1):
            self.edx = edx

            # go through the random indices in batches                
            for bottom_idx in range(0, self.num_samples, self.batch_size):
                # drop last batch if it's not a full batch
                if  self.num_samples - 1 - bottom_idx < self.batch_size:
                    break
                
                self.batch_indices = indices[bottom_idx : bottom_idx + self.batch_size]
                
                
                # make the batch of images and labels using the indices;
                self.collate_fn()
                
                # compute the class probabilities for each pixel;
                self.forward()
                
                # print intermediate metrics;
                if self.train_metrics:
                    self.metrics_step()
                    
                # compute the loss and update model parameters;
                self.update_model()

                self.scheduler.step()
                
            # reset steps; increment epoch; compute once-per-epoch functions and updates;
            self.end_of_epoch()
            
            # draw samples for next epoch, duplicates are allowed 
            indices = torch.multinomial(self.sample_probs, self.num_samples, replacement=True)
            
            
        print('Training done. Saving final model state..')
        self.save_states()
                
                
    def metrics_step(self):         
        with torch.no_grad():
            joints = self.embedding_space.get_joints(self.cprobs)
            preds = self.embedding_space.decide(joints)
            self.iou_fn.forward(preds, self.labels)
            self.acc_fn.forward(preds, self.labels)

    
    def print_intermediate(self, print_every=50):
        if self.global_steps % print_every == 0 and self.global_steps > 0:
            with torch.no_grad():
                avg_loss = str(round(self.running_loss / (self.steps + 1), 5))
                acc = str(round(self.acc_fn.compute().cpu().mean().item(), 5))
                miou = str(round(self.iou_fn.compute().cpu().mean().item(), 5))
                
                print("[current lr's]        {}".format([p['lr'] for p in self.optimizer.param_groups]))
                print('[global step]         {}'.format(str(self.global_steps)))
                print('[average loss]        {}'.format(avg_loss))
                print('[accuracy]            {}'.format(acc))
                print('[miou]                {}'.format(miou))
                                    
    
    def compute_metrics(self, mode):
        accuracy = self.acc_fn.compute().cpu()
        miou = self.iou_fn.compute().cpu()
        
        metrics = {'acc per class' : accuracy, 'miou per class' : miou}
        ncls = accuracy.size(0)
        
        self.print_metrics(metrics, ncls, mode)
        
        self.iou_fn.reset()
        self.acc_fn.reset()
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader, mode):
        with torch.no_grad():
            # steps are used to compute embedding norms every 15 steps
            self.steps = 0
            
            for images, labels, _ in loader:
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                self.forward()
                self.steps += 1
                self.metrics_step()
                
            self.compute_metrics(mode)


    def print_metrics(self, metrics, ncls, mode):
        print('-----------------[{} metrics epoch {}]-----------------'.format(mode, self.edx))
            
        print('\n[accuracy per class]')
        
        self.pretty_print([(self.i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ])
        
        print('\n[miou per class]')
        
        self.pretty_print([(self.i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ])
        
        print('[{} miou]     {}'.format(mode, metrics['miou per class'].mean().item()))
        
        print('[{} accuracy] {}'.format(mode, metrics['acc per class'].mean().item()))
        
        print('-----------------[end {} metrics epoch {}]-----------------\n\n'.format(mode, self.edx))
    
    
    def pretty_print(self, metrics_list):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            print(label, " " * offset, x)