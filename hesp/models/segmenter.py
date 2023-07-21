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
from hesp.util.norm_sampler import NormSampler
# from hesp.util.data_helpers import imshow

ROOT = '/home/mgonggri/master_thesis/'

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
        self.uncertainty_args = {}
        
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
        
            
    def forward(self):
        """Call embedding model, normalize embeddings and compute class probs from these."""
        
        # make the pixel embeddings with the backbone model;
        embs = self.embedding_model(self.images) 
        
        # perform multinomial logistic regression in the embedding space to compute the probs;
        self.proj_embs, self.cprobs = self.embedding_space(embs)
        
        # compute the norm weights 
        if self.config.segmenter._UNCERTAINTY != 'none':
            
            weight_args = {"embeddings"    : self.proj_embs,
                           "valid_mask"    : self.valid_mask,
                           "valid_labels"  : self.valid_labels,
                           "cond_probs"    : self.cprobs,
                           "method"        : self.config.segmenter._UNCERTAINTY}
            
            # compute the uncertainty based tensor for the loss; either a scalar or a set of weights depending on method   
            uncertainty = loss.compute_uncertainty_loss_augmentation(**weight_args)
            
            # store the uncertainty in kwargs for the loss.
            self.uncertainty_args = {self.config.segmenter._UNCERTAINTY : uncertainty}
            
        if self.config.segmenter._TRAIN_STOCHASTIC:
            # update the norm sampler with the new embedding norms
            self.norm_sampler.update_norms(self.proj_embs, self.valid_labels, self.valid_mask, self.batch_indices)
            # update the sample probs with the new norms
            self.norm_sampler.compute_sample_probs()
            
    def end_of_epoch(self):
        if self.config.segmenter._TRAIN_STOCHASTIC:
            # print the statistics of the sample probabilities
            print(
                '[min sample prob]',  self.norm_sampler.sample_probs.min().item(),   '\n'
                '[mean sample prob]', self.norm_sampler.sample_probs.mean().item(),  '\n'
                '[max sample prob]',  self.norm_sampler.sample_probs.max().item(),   '\n\n'
            )
            
            # print the statistics of the sample and class norms
            mean_sample_norms = self.norm_sampler.compute_mean_sample_norms()
            mean_class_norms = self.norm_sampler.compute_mean_class_norms()
            print('[mean sample norm min]', mean_sample_norms.min().item())
            print('[mean sample norm mean]', mean_sample_norms.mean().item())
            print('[mean sample norm max]', mean_sample_norms.max().item())
            print('[mean class norms] ', mean_class_norms.tolist(), '\n\n')
        
        self.embedding_space.running_norms = 0.0
        self.steps = 0
        self.running_loss = 0.0
        
        if self.train_metrics:
            self.compute_metrics('train')
            
        if self.val_metrics:
            self.compute_metrics_dataset(self.val_loader, 'val')
        
        self.steps = 0
        
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
        
        # compute the hierarchical cross entropy loss; kwargs may contain uncertainty tensors for loss augmentation 
        hce_loss = loss.CCE(
            valid_cprobs,
            self.valid_labels,
            self.tree,
            **self.uncertainty_args)
        
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
        
        self.acc_storage = []
        self.miou_storage = []
        self.norm_storage = []
        
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
            
            # initialize the norm sampler that stores the embedding norms and computes the sample probabilities
            self.norm_sampler = NormSampler(self.num_samples, self.config.dataset._NUM_CLASSES, device=self.device)
            
        else:
            self.num_samples = len(train.dataset)
            
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
    
    
    def train_fn_stochastic(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        ) -> None:
        
        """
        Probabilistic training loop that draws sample indices from a multinomial
        distribtution based on the inverse mean L2 norms of embeddings.
        """
        
        self.init_training_states(train_dataset, val_loader, optimizer, scheduler)
        torch.set_printoptions(sci_mode=False)
        
        for edx in range(self.start_edx, self.config.segmenter._NUM_EPOCHS, 1):
            self.edx = edx

            # go through the random indices in batches                
            for _ in range(0, self.num_samples, self.batch_size):
                
                # draw a batch of sample indeces from the distribution
                self.batch_indices = torch.multinomial(self.norm_sampler.sample_probs, self.batch_size, replacement=False)
                
                # make the batch of images and labels using the indices
                self.collate_fn()
                
                # compute the class probabilities for each pixel and update the sample probabilities
                self.forward()
                
                # print intermediate metrics
                if self.train_metrics:
                    self.metrics_step()
                    
                # compute the loss and update model parameters
                self.update_model()

                self.scheduler.step()
                
            # reset steps; increment epoch; compute once-per-epoch functions and updates
            self.end_of_epoch()
            
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