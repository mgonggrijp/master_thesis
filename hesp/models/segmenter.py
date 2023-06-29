from hesp.util.segmenter_helpers import *
from hesp.config.config import Config
from hesp.embedding_space.embedding_space import EmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.models.abstract_model import AbstractModel
from hesp.util import loss
from hesp.models.DeepLabV3Plus_Pytorch import network
import torchmetrics
import random
from hesp.util.data_helpers import imshow
from pprint import pprint

from time import sleep


class Segmenter(AbstractModel):
    def __init__(
            self,
            tree: Tree,
            config: Config,
            train_embedding_space: bool,
            prototype_path: str,
            device: torch.device,
            save_folder: str = "saves/",
            seed: float = None):

        self.save_folder = save_folder
        self.config = config
        self.tree = tree
        self.device = device
        self.seed = seed
        self.train_metrics = config.segmenter._TRAIN_METRICS
        self.val_metrics = config.segmenter._VAL_METRICS

        self.embedding_space = EmbeddingSpace(
            tree=tree,
            device=device,
            config=config,
            train=train_embedding_space,
            prototype_path=prototype_path)

        self.embedding_model = network.modeling._load_model(
            arch_type=config.base_model_name,
            backbone=config.segmenter._BACKBONE,
            num_classes=config.segmenter._EFN_OUT_DIM,
            output_stride=config.segmenter._OUTPUT_STRIDE,
            pretrained_backbone=config.segmenter._PRE_TRAINED_BB).to(device)

        if self.config.segmenter._ZERO_LABEL:
            raise NotImplementedError()
            # self.unseen_idxs = [tree.c2i[c]
            #                     for c in config.dataset._UNSEEN]

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
            validate_args=False).to(device)
        
        self.acc_fn = torchmetrics.classification.MulticlassAccuracy(
            config.dataset._NUM_CLASSES,
            average=None, 
            multidim_average='global',
            ignore_index=255,
            validate_args=False).to(device)
        
        self.recall_fn = torchmetrics.classification.MulticlassRecall(
            config.dataset._NUM_CLASSES,
            top_k=1,
            average=None,
            multidim_average='global',
            ignore_index=255,
            validate_args=False).to(device)
        
        
    def clip_norms(self):
        clip = self.config.segmenter._GRAD_CLIP
        
        # torch.nn.utils.clip_grad_norm_(
        #         [self.embedding_space.offsets,
        #          self.embedding_space.normals]\
        #         + list(self.embedding_model.parameters()), clip)
        
        torch.nn.utils.clip_grad_norm_(
            self.embedding_space.offsets, clip)
            
        torch.nn.utils.clip_grad_norm_(
            self.embedding_space.normals, clip)
        
        torch.nn.utils.clip_grad_norm_(
            self.embedding_model.parameters(), clip)
    
    
    def data_forward(self):
        embs = self.embedding_model(self.images)
        
        proj_embs = self.embedding_space.project(embs)
        
        logits = self.embedding_space.logits(
                    embeddings=proj_embs,
                    offsets=self.embedding_space.offsets,
                    normals=self.embedding_space.normals,
                    curvature=self.embedding_space.curvature)
        
        self.cprobs = self.embedding_space.softmax(logits)
  
        if self.computing_metrics or self.train_metrics:
            joints = self.embedding_space.get_joints(self.cprobs)
            preds = self.embedding_space.decide(joints)
            
            iou = self.iou_fn.forward(preds, self.labels)
            acc = self.acc_fn.forward(preds, self.labels)
            rec = self.recall_fn(preds, self.labels)
            
            if self.steps % 20 == 0:
                torch.set_printoptions(sci_mode=False)
                print('label unique', torch.unique(self.labels))
                print('label bins', torch.bincount(self.labels[self.labels < 255].flatten(), minlength=21))
                print('accuray', acc)       
                print('recall', rec)         
                print('\n\n\n')    
        
        
    def update_model(self):
        valid_mask = self.labels <= self.tree.M - 1
        valid_cprobs = self.cprobs.moveaxis(1, -1)[valid_mask]
        valid_labels = self.labels[valid_mask]
        hce_loss = loss.CCE(valid_cprobs, valid_labels, self.tree, self.steps)
        self.running_loss += hce_loss.item()
        self.clip_norms()
        hce_loss.backward()
        self.optimizer.step()
            

    def train(self, train_loader, val_loader, optimizer, scheduler):
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.computing_metrics = False
        
        if type(self.seed) == float:
            print('random seed', self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        print('training..')
        
        self.running_loss = 0.
        
        for edx in range(self.config.segmenter._NUM_EPOCHS):
            print('     [epoch]', edx)
            
            self.steps = 0
            
            for images, labels, _ in train_loader:
                self.optimizer.zero_grad()
                
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                
                # imshow(images.cpu(), labels.cpu())
                
                # for lab in labels:
                    # print( torch.bincount(lab[lab != 255].flatten()), '\n\n' )
                    # print(lab.unique(), '\n\n')
                # sleep(5)
                # print( labels )
                
                # compute the conditional probabilities from the images
                self.data_forward()
                
                self.steps += 1
                
                # update the model using the conditional probabilities and the labels
                self.update_model()
                

            self.steps = 0
            self.running_loss = 0.
            
            scheduler.step()
            
            # compute and print the metrics for the training data
            if self.train_metrics:
                print('----------------[Training Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics()
                print('----------------[End Training Metrics Epoch {}]----------------\n'.format(edx))
                
            # compute and print the metrics for the validation data
            if self.val_metrics:
                print('----------------[Validation Metrics Epoch {}]----------------\n'.format(edx))
                self.compute_metrics_dataset(val_loader)
                print('----------------[End Validation Metrics Epoch {}]----------------\n'.format(edx))
                
            
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
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader):
        
        with torch.no_grad():
            self.computing_metrics = True
            
            for images, labels, _ in loader:
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                self.data_forward()
                
            self.compute_metrics()
            self.computing_metrics = False


    def print_metrics(self, metrics, ncls, i2c):
        print('\n\n[accuracy per class]')
        self.pretty_print([(i2c[i], metrics['acc per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[miou per class]')
        self.pretty_print([(i2c[i], metrics['miou per class'][i].item()) for i in range(ncls) ])
        
        print('\n\n[recall per class]')
        self.pretty_print([(i2c[i], metrics['recall per class'][i].item()) for i in range(ncls) ])
        
        
    def pretty_print(self, metrics_list):
        target = 15
        for label, x in metrics_list:
            offset = target - len(label)
            print(label, " "*offset, x)
            