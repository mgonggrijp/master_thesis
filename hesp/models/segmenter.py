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
import logging




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
            logging.info('Loading embedding model and embedding space state dicts..')
            self.embedding_space.load_state_dict(
                torch.load(save_folder + "embedding_space.pt"))
            
            self.embedding_model.load_state_dict(
                torch.load(save_folder + "embedding_model.pt"))
            logging.info('Done.')
            
            
        self.iou_fn = torchmetrics.classification.MulticlassJaccardIndex(
            config.dataset._NUM_CLASSES, average=None, ignore_index=255, validate_args=False)
        
        self.acc_fn = torchmetrics.classification.MulticlassAccuracy(
            config.dataset._NUM_CLASSES, average=None,  multidim_average='global', ignore_index=255, validate_args=False)
        
        
    def max_sample_norm_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """ Normalize a batch of embeddings of shape (batch, dim, height, width) by the maximum norm over dim
        for every batch element and rescale them s.t. the maximum norm is equal to the radius 
        of the embedding space curvature: 1 / sqrt(c). """
        
        # sh (batch, heigth, width)
        norms = torch.linalg.vector_norm(embeddings, dim=1) 
        
        # sh (batch, )
        max_sample_norms = norms.amax(dim=(1, 2)) 
        
        # sh (batch, dim, height, width)
        normalized = embeddings / max_sample_norms[:, None, None, None] 

        # sh (batch, dim, height, width)
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
        
        if self.edx + 1 >= self.warmup_epochs:
            self.scheduler.step()
            logging.info('[new learning rate epoch {}]'.format(self.edx),
                    self.scheduler.get_last_lr())
        
        if self.computing_metrics:
            logging.info('----------------[Training Metrics Epoch {}]----------------\n'.format(self.edx))
            self.compute_metrics()
            logging.info('----------------[End Training Metrics Epoch {}]----------------\n'.format(self.edx))
            
        if self.val_metrics:
            logging.info('----------------[Validation Metrics Epoch {}]----------------\n'.format(self.edx))
            self.compute_metrics_dataset(self.val_loader)
            logging.info('----------------[End Validation Metrics Epoch {}]----------------\n'.format(self.edx))


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
        self.global_step += 1
       
        
    def warmup(self):
        """ Basic linear warmup scheduling. """
         # warmup schedule
        if self.edx < self.warmup_epochs: 
            for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.init_lrs[i] * (self.edx + 1) / self.warmup_epochs
                    logging.info('[new learning rate]', param_group['lr'])
     
            
    def train_fn(self, train_loader, val_loader, optimizer, scheduler, warmup_epochs):
        self.init_training_states(train_loader, val_loader, optimizer, scheduler, warmup_epochs)
        
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.basicConfig(filename=self.config.segmenter._SAVE_FOLDER + 'output.log', level=logging.INFO)
        
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            logging.info('     [epoch]', edx)
            self.steps = 0
            self.edx = edx
            
            # if edx + 1 <= warmup_epochs take a warming up step
            self.warmup()
            
            for images, labels, _ in train_loader:
                self.labels = labels.to(self.device).squeeze()
                self.images = images.to(self.device)
                
                # compute the class probabilities for each pixel;
                self.forward()
                
                # print intermediate metrics;
                if self.computing_metrics:
                    self.metrics_step()
                
                # compute the loss and update model parameters;
                self.update_model()

            # reset steps, increment epoch, take a scheduler step and 
            self.end_of_epoch()
            
        logging.info('Training done. Saving final model state..')
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
        self.computing_metrics = True
        self.running_loss = 0.0
        self.global_step = 0
        self.steps = 0
        
        if type(self.seed) == float:
            logging.info('[set random seed]  ', self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            
        self.init_lrs = []
        for param_group in self.optimizer.param_groups:
            self.init_lrs.append(param_group['lr'])
        logging.info(self.init_lrs)
    
    
    def train_fn_stochastic(self, train_dataset, val_loader, optimizer, scheduler, warmup_epochs):
        """ Probabilistic training loop that draws sample indeces from a multinomial distribtution. """
        logging.info("Starting stochastic training...")
        self.init_training_states(train_dataset, val_loader, optimizer, scheduler, warmup_epochs)
        
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.basicConfig(filename=self.config.segmenter._SAVE_FOLDER + 'output.log', level=logging.INFO)
        
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
                if self.computing_metrics:
                    self.metrics_step()
                
                # compute the loss and update model parameters;
                self.update_model()

            # reset steps, increment epoch, take a scheduler step and 
            self.end_of_epoch()
            
        logging.info('Training done. Saving final model state..')
        self.save_states()
                
                
    def metrics_step(self):           
        with torch.no_grad():
            joints = self.embedding_space.get_joints(self.cprobs)
            preds = self.embedding_space.decide(joints)
            iou = self.iou_fn.forward(preds, self.labels)
            acc = self.acc_fn.forward(preds, self.labels)

    
    def print_intermediate(self, print_every=50):
        
        if self.steps % print_every == 0 and self.steps > 0:
            
            with torch.no_grad():
                accuracy = self.acc_fn.compute().cpu().mean().item()
                miou = self.iou_fn.compute().cpu().mean().item()
                logging.info('[global step]         ', round(self.global_step, 5))
                logging.info('[average loss]        ', round(self.running_loss / (self.steps + 1), 5))
                logging.info('[accuracy]            ', round(accuracy, 5))
                logging.info('[miou]                ', round(miou, 5))
                
                offset_norms_1 = torch.linalg.vector_norm(self.embedding_space.offsets, dim=1).mean().item()
                normal_norms_1 = torch.linalg.vector_norm(self.embedding_space.normals, dim=1).mean().item()
                
                logging.info('[offset norms dim 1] ', round(offset_norms_1, 8) )
                logging.info('[normal norm dim 1]  ', round(normal_norms_1, 8),  '\n\n')
    
    
    def compute_metrics(self):
        
        if self.config.dataset._NAME == 'pascal':
            i2c_file = "datasets/pascal/PASCAL_i2c.txt"
        
        with open(i2c_file, "r") as f:
            i2c = {i : line.split(":")[1][:-1] for i, line in enumerate(f.readlines()) }
        
        accuracy = self.acc_fn.compute().cpu()
        miou = self.iou_fn.compute().cpu()
        
        metrics = {'acc per class' : accuracy,
                   'miou per class' : miou}
        
        ncls = accuracy.size(0)
        
        self.print_metrics(metrics, ncls, i2c)

        self.iou_fn.reset()
        self.acc_fn.reset()
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader):
        with torch.no_grad():
            for images, labels, _ in loader:
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                
                self.forward()
                
                self.metrics_step()
                
            self.compute_metrics()


    def print_metrics(self, metrics, ncls, i2c):
        # logging.info('\n\n[accuracy per class]')
        # self.pretty_logging.info([(i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ])
        # logging.info('\n\n[miou per class]')
        # self.pretty_logging.info([(i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ])
        
        logging.info('\n\n[global step]       ', self.global_step,
                '\n[miou]              ', metrics['miou per class'].mean().item(), 
                '\n[average accuracy]  ', metrics['acc per class'].mean().item()) 
        
        
    def pretty_print(self, metrics_list):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            logging.info(label, " "*offset, x)
            