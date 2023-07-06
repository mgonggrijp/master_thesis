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
# from hesp.util.data_helpers import imshow


with open("root_folder.txt", "r") as f:
    lines = f.readlines()
    ROOT = lines[0]
    

class Segmenter(torch.nn.Module):
    def __init__(
        self,
        tree: Tree,
        config: Config,
        device,
        save_folder: str = "saves/",
        seed: float = None
    ):        
        
        super().__init__()
        
        self.save_folder = save_folder
        self.config = config
        self.tree = tree
        self.seed = seed
        self.train_metrics = config.segmenter._TRAIN_METRICS    
        self.val_metrics = config.segmenter._VAL_METRICS
        self.device = device
        self.save_state = config.segmenter._SAVE_STATE
        
        if self.config.dataset._NAME == 'pascal':
            i2c_file = config._ROOT + "datasets/pascal/PASCAL_i2c.txt"
            with open(i2c_file, "r") as f:
                self.i2c = {i : line.split(":")[1][:-1] for i, line in enumerate(f.readlines()) }
            
        if config.embedding_space._GEOMETRY == 'hyperbolic':
            self.embedding_space = HyperbolicEmbeddingSpace(tree, config)
        elif config.embedding_space._GEOMETRY == 'euclidean':
            self.embedding_space = EuclideanEmbeddingSpace(tree, config)
            
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
        
        
        
        
    def max_sample_norm_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """ 
        Normalize a batch of embeddings of shape (batch, dim, height, width) by the maximum norm over dim
        for every batch element and rescale them s.t. the maximum norm is equal to the radius 
        of the embedding space curvature: 1 / sqrt(c).
        """
        norms = torch.linalg.vector_norm(embeddings, dim=1) 
        max_sample_norms = norms.amax(dim=(1, 2)) 
        normalized = embeddings / max_sample_norms[:, None, None, None] 
        rescaled_normalized = normalized * (1.0 / torch.sqrt(self.embedding_space.curvature))
        return rescaled_normalized
            
            
            
    
    def forward(self):
        """
        Call embedding model, normalize embeddings and compute class probs from these.
        """
        # sh (batch, dim, height, width)
        embs = self.embedding_model(self.images) 
        # normalize the norms in each sample by the maximum norm for that same sample;
        normalized = self.max_sample_norm_normalize(embs) # (batch, dim, height, width)
        # sh (batch, nclasses, height, width)
        self.cprobs = self.embedding_space(normalized, self.steps) 

    
    
    
    def end_of_epoch(self):
        average_train_norms = self.embedding_space.running_norms / self.steps
        print('[average train embedding norms]  {}'.format(str(average_train_norms)))
        
        self.embedding_space.running_norms = 0.0
        self.steps = 0
        self.running_loss = 0.0
        
        if self.edx + 1 >= self.warmup_epochs and not self.config.segmenter._CYCLIC:
            print('scheduler step...')
            self.scheduler.step()
            
        if self.train_metrics:
            self.compute_metrics('train')
            
        if self.val_metrics:
            self.compute_metrics_dataset(self.val_loader, 'val')
        
        average_val_norms = self.embedding_space.running_norms / self.steps
        print('[average val embedding norms]    {}'.format(str(average_val_norms)))
        
        self.embedding_space.running_norms = 0.0
        self.steps = 0

        

        
    def update_model(self):
        """ Compute the loss based on the probs and labels. Clip grads, loss backwards and optimizer step. """
        
        valid_mask = self.labels <= self.tree.M - 1
        valid_cprobs = self.cprobs.moveaxis(1, -1)[valid_mask]
        valid_labels = self.labels[valid_mask]
        
        hce_loss = loss.CCE(valid_cprobs, valid_labels, self.tree, self.class_weights)
        self.running_loss += hce_loss.item()
        
        self.print_intermediate()
        
        torch.nn.utils.clip_grad_norm_(
            self.embedding_space.offsets,
            self.config.segmenter._GRAD_CLIP
        )
        
        hce_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.steps += 1
        self.global_steps += 1
       
       
        
       
        
    def warmup(self):
        """ Basic linear warmup scheduling. """
        print("warmup step...")
        for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.init_lrs[i] * (self.edx + 1) / self.warmup_epochs
                print(param_group['lr'])    
                
        
                
     
    def train_fn(self, train_loader, val_loader, optimizer, scheduler, warmup_epochs):
        
        self.init_training_states(train_loader, val_loader, optimizer, scheduler, warmup_epochs)

        for edx in range(self.start_edx, self.config.segmenter._NUM_EPOCHS + self.start_edx, 1):
            self.edx = edx
            
            # if still in warmup epochs, take warmup steps
            if self.edx < self.warmup_epochs:
                self.warmup()
            
            for images, labels, _ in train_loader:
                self.labels = labels.to(self.device).squeeze()
                self.images = images.to(self.device)
                
                # compute the class probabilities for each pixel
                self.forward()
                
                # compute the loss and update model parameters
                self.update_model()
                
                 # when using cyclic learning rate scheduling
                if self.config.segmenter._CYCLIC:
                    self.scheduler.step()  
                
                # print intermediate metrics
                if self.train_metrics:
                    self.metrics_step()
                
            # reset steps, increment epoch, take a scheduler step, and update parameters
            self.end_of_epoch()
            
        if self.save_state:
            self.save_states()
            print('Training done. Saving final model state..')
    
    
    
            
    def save_states(self):
        save_folder = self.config.segmenter._SAVE_FOLDER
        
        print(save_folder)
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        torch.save(self.embedding_model.state_dict(), save_folder + 'embedding_model.pt')
        torch.save(self.embedding_space.state_dict(), save_folder + 'embedding_space.pt')
        torch.save(self.scheduler.state_dict(), save_folder + 'scheduler.pt')
        torch.save(self.optimizer.state_dict(), save_folder + 'optimizer.pt')
        
        with open(save_folder + 'epoch.txt', 'w') as f:
            f.write("{}\n{}".format(str(self.edx + 1), str(self.global_steps)))


            
            
    def collate_fn(self):
        """
        Given a dataset and a list of indeces return a batch of samples.
        Returns a tuple of (batch_samples, batch_labels)
        """
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
        self.class_weights = torch.load(ROOT + "datasets/pascal/class_weights.pt").to(self.device)
        self.warmup_epochs = warmup_epochs
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
    
    
    
    
    def train_fn_stochastic(self, train_dataset, val_loader, optimizer, scheduler, warmup_epochs):
        """ Probabilistic training loop that draws sample indeces from a multinomial distribtution. """
        self.init_training_states(train_dataset, val_loader, optimizer, scheduler, warmup_epochs)
        torch.set_printoptions(sci_mode=False)
        
        for edx in range(self.start_edx, self.config.segmenter._NUM_EPOCHS, 1):
            self.edx = edx
            
            # generate all sample indeces for this epoch; allow for doubles
            indeces = torch.multinomial(self.sample_probs, self.num_samples, replacement=True)
            
            # warmup schedule
            self.warmup()
            
            for i in range(0, self.num_samples, self.batch_size):
                bottom, upper = i, i + self.batch_size
                
                # drop last batch if it becomes smaller than 2 samples;
                if self.num_samples - i < 2:
                    break
                
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
            self.iou_fn.forward(preds, self.labels)
            self.acc_fn.forward(preds, self.labels)


    
    
    def print_intermediate(self, print_every=50):
        if self.steps % print_every == 0 and self.steps > 0:
            with torch.no_grad():
                avg_loss = str(round(self.running_loss / (self.steps + 1), 5))
                acc = str(round(self.acc_fn.compute().cpu().mean().item(), 5))
                miou = str(round(self.iou_fn.compute().cpu().mean().item(), 5))
                
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