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
                torch.load(save_folder + "embedding_space.pt"))
            
            self.embedding_model.load_state_dict(
                torch.load(save_folder + "embedding_model.pt"))
            print('Done.')
            
            
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
            
    
    def forward(self, images):
        
        # sh (batch, dim, height, width)
        embs = self.embedding_model(images) 
        
        # normalize the norms in each sample by the maximum norm for that same sample;
        # norms are rescaled by 1 / sqrt(c);
        normalized = self.max_sample_norm_normalize(embs) # (batch, dim, height, width)
   
        # sh (batch, nclasses, height, width)
        cprobs = self.embedding_space(normalized, self.steps) 
        
        return cprobs

            
    def train_fn(self, train_loader, val_loader, optimizer, scheduler, warmup_epochs):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.computing_metrics = True
        self.train()
        
        if type(self.seed) == float:
            print('[random seed]  ', self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        init_lrs = []
        for param_group in optimizer.param_groups:
            init_lrs.append( param_group['lr'])
        print(init_lrs)
        
        self.running_loss = 0.
        self.global_step = 0
        torch.set_printoptions(sci_mode=False)
        
        class_weights = torch.load("datasets/pascal/class_weights.pt").to(self.device)
        
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            print('     [epoch]', edx)
            self.steps = 0
            
            # warmup schedule
            if edx + 1 <= warmup_epochs: 
                for i, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = init_lrs[i] * (edx + 1) / warmup_epochs
                        print('[new learning rate]', param_group['lr'])
            
            for images, labels, _ in train_loader:
                labels = labels.to(self.device).squeeze()
                images = images.to(self.device)
                
                cprobs = self.forward(images)
                
                if self.computing_metrics:
                    self.metrics_step(cprobs, labels)
                
                valid_mask = labels <= self.tree.M - 1
                
                valid_cprobs = cprobs.moveaxis(1, -1)[valid_mask]
                
                valid_labels = labels[valid_mask]
                
                hce_loss = loss.CCE(valid_cprobs, valid_labels, self.tree, class_weights)
                
                self.running_loss += hce_loss.item()
                
                if self.steps % 50 == 0 and self.steps > 0:
                    
                    with torch.no_grad():
                        accuracy = self.acc_fn.compute().cpu().mean().item()
                        miou = self.iou_fn.compute().cpu().mean().item()
                        print('[global step]         ', round(self.global_step, 5))
                        print('[average loss]        ', round(self.running_loss / (self.steps + 1), 5))
                        print('[accuracy]            ', round(accuracy, 5))
                        print('[miou]                ', round(miou, 5))
                        
                        offset_norms_1 = torch.linalg.vector_norm(self.embedding_space.offsets, dim=1).mean().item()
                        normal_norms_1 = torch.linalg.vector_norm(self.embedding_space.normals, dim=1).mean().item()
                        
                        print('[offset norms dim 1] ', round(offset_norms_1, 8) )
                        print('[normal norm dim 1]  ', round(normal_norms_1, 8),  '\n\n')
                    
                torch.nn.utils.clip_grad_norm_(
                    self.embedding_space.offsets, self.config.segmenter._GRAD_CLIP)
                
                hce_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.steps += 1
                self.global_step += 1

            self.steps = 0
            self.running_loss = 0.
            
            if edx + 1 >= warmup_epochs:
                scheduler.step()
                print('[new learning rate epoch {}]'.format(edx),
                      scheduler.get_last_lr())
            
            if self.computing_metrics:
                print('----------------[Training Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics()
                print('----------------[End Training Metrics Epoch {}]----------------\n'.format(edx))
                
            if self.val_metrics:
                print('----------------[Validation Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics_dataset(val_loader)
                print('----------------[End Validation Metrics Epoch {}]----------------\n'.format(edx))
    
        print('Training done. Saving final model state..')
        folder =  self.config.segmenter._SAVE_FOLDER 
        if not os.path.exists(folder):
            os.mkdir(folder)
        # torch.save(self.embedding_model.state_dict(), folder + 'segmenter.pt' )
        torch.save(self.embedding_model.state_dict(), folder + 'embedding_model.pt' )
        torch.save(self.embedding_space.state_dict(), folder + 'embedding_space.pt' )
        
    
    def metrics_step(self, cprobs, labels):           
        with torch.no_grad():
            joints = self.embedding_space.get_joints(cprobs)
            preds = self.embedding_space.decide(joints)
            iou = self.iou_fn.forward(preds, labels)
            acc = self.acc_fn.forward(preds, labels)

    
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
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                cprobs = self.forward(images)
                self.metrics_step(cprobs, labels)
            self.compute_metrics()


    def print_metrics(self, metrics, ncls, i2c):
        # print('\n\n[accuracy per class]')
        # self.pretty_print([(i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ])
        # print('\n\n[miou per class]')
        # self.pretty_print([(i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[global step]       ', self.global_step,
                '\n[miou]              ', metrics['miou per class'].mean().item(), 
                '\n[average accuracy]  ', metrics['acc per class'].mean().item()) 
        
        
    def pretty_print(self, metrics_list):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            print(label, " "*offset, x)
            