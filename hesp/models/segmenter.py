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

        # continue training using previous model state and embedding space offsets / normals
        if config.segmenter._RESUME:
            self.embedding_model.load_state_dict(
                torch.load(self.save_folder + 'model.pt'))
            
            self.embedding_space.normals = torch.load(
                self.save_folder + "normals.pt").to(self.device)
            
            self.embedding_space.offsets = torch.load(
                self.save_folder + "offsets.pt").to(self.device)
            
        self.iou_fn = torchmetrics.classification.MulticlassJaccardIndex(
            config.dataset._NUM_CLASSES,
            average=None,
            ignore_index=255,
            validate_args=False)
        
        self.acc_fn = torchmetrics.classification.MulticlassAccuracy(
            config.dataset._NUM_CLASSES,
            average=None, 
            multidim_average='global',
            ignore_index=255,
            validate_args=False)
        
        self.recall_fn = torchmetrics.classification.MulticlassRecall(
            config.dataset._NUM_CLASSES,
            top_k=1,
            average=None,
            multidim_average='global',
            ignore_index=255,
            validate_args=False)
        
    
    def forward(self, images):
        embs = self.embedding_model(images)
        cprobs = self.embedding_space(embs)        
        return cprobs

            
    def train_fn(self, train_loader, val_loader, optimizer, scheduler, warmup_scheduler, warmup_epochs):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.computing_metrics = True
        self.train()
        
        if type(self.seed) == float:
            print('random seed', self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        print('training..')
        
        self.running_loss = 0.
        self.global_step = 0
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            print('     [epoch]', edx)
            self.steps = 0
            
            if edx > warmup_epochs:
                print('[learning rate]', scheduler.get_last_lr())
                
            else:
                print('[learning rate]', warmup_scheduler.get_last_lr())
            
            
            for images, labels, _ in train_loader:
                labels = labels.to(self.device).squeeze()
                images = images.to(self.device)
                
                cprobs = self.forward(images)
                
                if self.computing_metrics:
                    self.metrics_step(cprobs, labels)
                
                valid_mask = labels <= self.tree.M - 1
                
                valid_cprobs = cprobs.moveaxis(1, -1)[valid_mask]
                
                valid_labels = labels[valid_mask]
                
                hce_loss = loss.CCE(valid_cprobs, valid_labels, self.tree, self.steps)
                
                self.running_loss += hce_loss.item()
                
                if self.steps % 2 == 0 and self.steps > 0:
                    accuracy = self.acc_fn.compute().cpu().mean().item()
                    miou = self.iou_fn.compute().cpu().mean().item()
                    recall = self.recall_fn.compute().cpu().mean().item()
                    print('[global step]  ', self.global_step)
                    print('[average loss] ', self.running_loss / (self.steps + 1))
                    print('[accuracy]     ', accuracy)
                    print('[miou]         ', miou)
                    print('[recall]       ', recall)
                    print('[learning rate]', optimizer)
                    
                
                torch.nn.utils.clip_grad_norm_(
                    self.embedding_space.offsets,
                    self.config.segmenter._GRAD_CLIP)
                
                hce_loss.backward()
                
                self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                self.steps += 1
                self.global_step += 1

            self.steps = 0
            self.running_loss = 0.
            
            if edx > warmup_epochs - 1:
                scheduler.step()
                print('scheduler step')
            else:
                warmup_scheduler.step()
                print('warmup scheduler step')
            
            
            # compute and print the metrics for the training data
            if self.computing_metrics:
                print('----------------[Training Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics()
                print('----------------[End Training Metrics Epoch {}]----------------\n'.format(edx))
                
            # compute and print the metrics for the validation data
            if self.val_metrics:
                print('----------------[Validation Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics_dataset(val_loader)
                print('----------------[End Validation Metrics Epoch {}]----------------\n'.format(edx))
    
    
    def metrics_step(self, cprobs, labels):           
        with torch.no_grad():
            joints = self.embedding_space.get_joints(cprobs)
            preds = self.embedding_space.decide(joints)
            iou = self.iou_fn.forward(preds, labels)
            acc = self.acc_fn.forward(preds, labels)
            rec = self.recall_fn(preds, labels)        

    
    def compute_metrics(self):
        
        if self.config.dataset._NAME == 'pascal':
            i2c_file = "datasets/pascal/PASCAL_i2c.txt"
        
        with open(i2c_file, "r") as f:
            i2c = {i : line.split(":")[1][:-1] for i, line in enumerate(f.readlines()) }
        
        accuracy = self.acc_fn.compute().cpu()
        miou = self.iou_fn.compute().cpu()
        recall = self.recall_fn.compute().cpu()
        
        metrics = {'acc per class' : accuracy,
                   'miou per class' : miou,
                   'recall per class' : recall}
        
        ncls = accuracy.size(0)
        
        self.print_metrics(metrics, ncls, i2c)

        self.iou_fn.reset()
        self.acc_fn.reset()
        self.recall_fn.reset()
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader):
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                cprobs = self.forward(images)
                self.metrics_step(cprobs, labels)
            self.compute_metrics()


    def print_metrics(self, metrics, ncls, i2c):
        print('\n\n[accuracy per class]')
        self.pretty_print([(i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[miou per class]')
        self.pretty_print([(i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[recall per class]')
        self.pretty_print([(i2c[i], metrics['recall per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[Global Step]       ', self.global_step,
                '\n[Average MIOU]      ', metrics['miou per class'].mean().item(), 
                '\n[Average Accuracy]  ', metrics['acc per class'].mean().item(), 
                '\n[Average Recall]    ', metrics['recall per class'].mean().item())
        
    def pretty_print(self, metrics_list):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            print(label, " "*offset, x)
            