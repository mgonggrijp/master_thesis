from hesp.util.segmenter_helpers import *
from hesp.config.config import Config
from hesp.embedding_space.hyperbolic_embedding_space import HyperbolicEmbeddingSpace
from hesp.embedding_space.euclidean_embedding_space import EuclideanEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util import loss
from hesp.models.DeepLabV3Plus_Pytorch import network
import torchmetrics
import random
from hesp.util.data_helpers import imshow
import os


class Segmenter(torch.nn.Module):
    def __init__(self, tree: Tree, config: Config, device, save_folder: str = "saves/", seed: float = None):
        super().__init__()
        self.save_folder = save_folder
        self.config = config
        self.tree = tree
        self.seed = seed
        self.train_metrics = config.segmenter._TRAIN_METRICS    
        self.val_metrics = config.segmenter._VAL_METRICS
        self.device = device
        if self.config.dataset._NAME == 'pascal':
            i2c_file = "datasets/pascal/PASCAL_i2c.txt"
        with open(i2c_file, "r") as f:
            self.i2c = {i : line.split(":")[1][:-1] for i, line in enumerate(f.readlines()) }
        if config.embedding_space._GEOMETRY == 'hyperbolic':
            self.embedding_space = HyperbolicEmbeddingSpace(tree, config)
        if config.embedding_space._GEOMETRY == 'euclidean':
            self.embedding_space = EuclideanEmbeddingSpace(tree, config)
        self.embedding_model = network.modeling._load_model(
            arch_type=config.base_model_name,
            backbone=config.segmenter._BACKBONE,
            num_classes=config.segmenter._EFN_OUT_DIM,
            output_stride=config.segmenter._OUTPUT_STRIDE,
            pretrained_backbone=config.segmenter._PRE_TRAINED_BB)
        if self.config.segmenter._RESUME:
            print('Loading embedding model and embedding space state dicts..')
            self.embedding_space.load_state_dict(
                torch.load(save_folder + "embedding_space.pt"), file=f)
            self.embedding_model.load_state_dict(
                torch.load(save_folder + "embedding_model.pt"), file=f)
            print('Done.')
        self.iou_fn = torchmetrics.classification.MulticlassJaccardIndex(
            config.dataset._NUM_CLASSES, average=None, ignore_index=255, validate_args=False)
        self.acc_fn = torchmetrics.classification.MulticlassAccuracy(
            config.dataset._NUM_CLASSES, average=None,  multidim_average='global', ignore_index=255, validate_args=False)
        
        
    def max_sample_norm_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """ Normalize a batch of embeddings of shape (batch, dim, height, width) by the maximum norm over dim
        for every batch element and rescale them s.t. the maximum norm is equal to the radius 
        of the embedding space curvature: 1 / sqrt(c). """
        norms = torch.linalg.vector_norm(embeddings, dim=1) 
        max_sample_norms = norms.amax(dim=(1, 2)) 
        normalized = embeddings / max_sample_norms[:, None, None, None] 
        rescaled_normalized = normalized * ( 1.0 / torch.sqrt(self.embedding_space.curvature) )
        return rescaled_normalized
            
    
    def forward(self):
        # sh (batch, dim, height, width)
        embs = self.embedding_model(self.images) 
        # normalize the norms in each sample by the maximum norm for that same sample;
        # norms are rescaled by 1 / sqrt(c);
        normalized = self.max_sample_norm_normalize(embs) # (batch, dim, height, width)
        # sh (batch, nclasses, height, width)
        self.cprobs = self.embedding_space(normalized, self.steps) 

    
    def end_of_epoch(self):
        self.steps = 0
        self.running_loss = 0.
        edx = str(self.edx)
        if self.edx + 1 >= self.warmup_epochs:
            self.scheduler.step()
        with open(self.config.segmenter._SAVE_FOLDER + 'output.txt', 'a') as f:
            if self.train_metrics:
                self.compute_metrics('train')
            if self.val_metrics:
                self.compute_metrics_dataset(self.val_loader, 'val')


    def update_model(self):
        valid_mask = self.labels <= self.tree.M - 1
        valid_cprobs = self.cprobs.moveaxis(1, -1)[valid_mask]
        valid_labels = self.labels[valid_mask]
        hce_loss = loss.CCE(valid_cprobs, valid_labels, self.tree, self.class_weights)
        self.running_loss += hce_loss.item()
        self.print_intermediate()
        torch.nn.utils.clip_grad_norm_(
            self.embedding_space.offsets,
            self.config.segmenter._GRAD_CLIP)
        hce_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.steps += 1
        self.global_steps += 1
       
        
    def warmup(self):
        """ Basic linear warmup scheduling. """
         # warmup schedule
        if self.edx < self.warmup_epochs: 
            for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.init_lrs[i] * (self.edx + 1) / self.warmup_epochs
     
            
    def train_fn(self, train_loader, val_loader, optimizer, scheduler, warmup_epochs):
        self.init_training_states(train_loader, val_loader, optimizer, scheduler, warmup_epochs)
        
        
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            self.edx = edx
            # if still in warmup epochs, take warmup steps
            self.warmup()
            for images, labels, _ in train_loader:
                self.labels = labels.to(self.device).squeeze()
                self.images = images.to(self.device)
                # compute the class probabilities for each pixel;
                self.forward()
                # print intermediate metrics;
                if self.train_metrics:
                    self.metrics_step()
                # compute the loss and update model parameters;
                self.update_model()
            # reset steps, increment epoch, take a scheduler step and update parameters
            self.end_of_epoch()
            
        with open(self.config.segmenter._SAVE_FOLDER + 'output.txt', 'a') as f:
           print('Training done. Saving final model state..', file=f)
           
        self.save_states()
    
    
    def save_states(self):
        folder =  self.config.segmenter._SAVE_FOLDER 
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.embedding_model.state_dict(), folder + 'embedding_model.pt' )
        torch.save(self.embedding_space.state_dict(), folder + 'embedding_space.pt' )
    
    
    def collate_fn(self):
        """ Given a dataset and a list of indeces return a batch of samples.
        Returns a tuple of (batch_samples, batch_labels) """
        image_batch = []
        label_batch = []
        for idx in self.batch_indeces:
            samples, labels, _ = self.dataset[idx]
            image_batch.append(samples[None, :])
            label_batch.append(labels)
        self.images = torch.cat(image_batch).to(self.device)
        self.labels = torch.cat(label_batch).to(self.device).squeeze()
        
        
    def init_training_states(self, train, val_loader, optimizer, scheduler, warmup_epochs):
        """ Initialize all the variables into self which are used for training. """
        if self.config.segmenter._TRAIN_STOCHASTIC:
            self.dataset = train
            self.num_samples = len(self.dataset)
            self.batch_size = self.config.segmenter._BATCH_SIZE
            self.sample_probs = torch.ones(self.num_samples) / self.num_samples
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.class_weights = torch.load("datasets/pascal/class_weights.pt").to(self.device)
        self.warmup_epochs = warmup_epochs
        self.running_loss = 0.0
        self.global_steps = 0
        self.steps = 0
        if type(self.seed) == float:
            with open(self.config.segmenter._SAVE_FOLDER + 'output.txt', 'a') as f:
                print('[random seed]  ', self.seed, file=f)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
        self.init_lrs = []
        for param_group in self.optimizer.param_groups:
            self.init_lrs.append(param_group['lr'])
    
    
    def train_fn_stochastic(self, train_dataset, val_loader, optimizer, scheduler, warmup_epochs):
        """ Probabilistic training loop that draws sample indeces from a multinomial distribtution. """
        self.init_training_states(train_dataset, val_loader, optimizer, scheduler, warmup_epochs)
        torch.set_printoptions(sci_mode=False)
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            self.edx = edx
            # generate all sample indeces for this epoch; allow for doubles
            indeces = torch.multinomial(self.sample_probs, self.num_samples, replacement=True)
            # warmup schedule
            self.warmup()
            for i in range(0, self.num_samples, self.batch_size):
                # get slices of batch size for the sample indeces
                bottom, upper = i, i + self.batch_size
                # drop last batch if it becomes smaller than 2 samples;
                if self.num_samples - i < 2:
                    break
                # get the indeces for current batch;
                self.batch_indeces = indeces[bottom : upper]
                # make the batch of images and labels using the indeces;
                self.collate_fn()
                # compute the class probabilities for each pixel;
                self.forward()
                # print intermediate metrics;
                if self.train_metrics:
                    self.metrics_step()
                # compute the loss and update model parameters;
                self.update_model()
            # reset steps, increment epoch, take a scheduler step and 
            self.end_of_epoch()
        print('Training done. Saving final model state..')
        self.save_states()
                
                
    def metrics_step(self):           
        with torch.no_grad():
            joints = self.embedding_space.get_joints(self.cprobs)
            preds = self.embedding_space.decide(joints)
            iou = self.iou_fn.forward(preds, self.labels)
            acc = self.acc_fn.forward(preds, self.labels)

    
    def print_intermediate(self, print_every=5):
        if self.steps % print_every == 0 and self.steps > 0:
            with torch.no_grad():
                offsn1 = str(round(torch.linalg.vector_norm(self.embedding_space.offsets, dim=1).mean().item(), 5))
                normn1 = str(round(torch.linalg.vector_norm(self.embedding_space.normals, dim=1).mean().item(), 5))
                avg_loss = str(round(self.running_loss / (self.steps + 1), 5))
                acc = str(round(self.acc_fn.compute().cpu().mean().item(), 5))
                miou = str(round(self.iou_fn.compute().cpu().mean().item(), 5))
                with open(self.config.segmenter._SAVE_FOLDER + 'output.txt', 'a') as f:
                    print('[global step]         {}'.format(str(self.global_steps)), file=f)
                    print('[average loss]        {}'.format(avg_loss), file=f)
                    print('[accuracy]            {}'.format(acc), file=f)
                    print('[miou]                {}'.format(miou), file=f)
                    print('[normal norm dim 1]   {}'.format(normn1), file=f)
                    print('[offset norms dim 1]  {}\n\n'.format(offsn1),  file=f)
                                     
    
    def compute_metrics(self, mode):
        accuracy = self.acc_fn.compute().cpu()
        miou = self.iou_fn.compute().cpu()
        metrics = {'acc per class' : accuracy,
                   'miou per class' : miou}
        ncls = accuracy.size(0)
        self.print_metrics(metrics, ncls, mode)
        self.iou_fn.reset()
        self.acc_fn.reset()
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader, mode):
        with torch.no_grad():
            for images, labels, _ in loader:
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                self.forward()
                self.metrics_step()
            self.compute_metrics(mode)


    def print_metrics(self, metrics, ncls, mode):
        with open(self.config.segmenter._SAVE_FOLDER + 'output.txt', 'a') as f:
            
            if mode == 'train':
                print('-----------------[Training Metrics Epoch {}]-----------------'.format(self.edx), file=f)
                
            if mode == 'val':
                print(['-----------------[Validation Metrics Epoch {}]-----------------'.format(self.edx)], file=f)
                
            print('\n\n[accuracy per class]', file=f)
            
            self.pretty_print([(self.i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ], f)
            
            print('\n\n[miou per class]', file=f)
            
            self.pretty_print([(self.i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ], f)
            
            print('\n[miou]              ', metrics['miou per class'].mean().item(), 
                  '\n[average accuracy]  ', metrics['acc per class'].mean().item(),
                   file=f)
            
            if mode == 'train':
                print('-----------------[End Training Metrics Epoch {}]-----------------'.format(self.edx), file=f)
                
            if mode == 'val':
                print(['-----------------[End Validation Metrics Epoch {}]-----------------'.format(self.edx)], file=f)
        
        
    def pretty_print(self, metrics_list, f):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            print(label, " "*offset, x, file=f)
            