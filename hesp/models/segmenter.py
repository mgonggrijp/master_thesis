from hesp.util.segmenter_helpers import *
from hesp.config.config import Config
from hesp.embedding_space.embedding_space import EmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.models.abstract_model import AbstractModel
from hesp.util import loss
from hesp.models.DeepLabV3Plus_Pytorch import network
import torchmetrics
import random


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
        
        
    def clip_norms(self):
        clip = self.config.segmenter._GRAD_CLIP
        
        torch.nn.utils.clip_grad_norm_(
                [self.embedding_space.offsets,
                 self.embedding_space.normals]\
                + list(self.embedding_model.parameters()), clip)
        
        # torch.nn.utils.clip_grad_norm_(
            # self.embedding_space.offsets, clip)
            
        # torch.nn.utils.clip_grad_norm_(
            # self.embedding_space.normals, clip)
        # 
        # torch.nn.utils.clip_grad_norm_(
            # self.embedding_model.parameters(), clip)
    
    
    def data_forward(self):
        embs = self.embedding_model(self.images)
        
        proj_embs = self.embedding_space.project(embs)
        
        logits = self.embedding_space.logits(
                    embeddings=proj_embs,
                    offsets=self.embedding_space.offsets,
                    normals=self.embedding_space.normals,
                    curvature=self.embedding_space.curvature)
        
        max_value, max_index = torch.max(logits, dim=1)

        self.cprobs = self.embedding_space.softmax(logits)
        
        if self.steps % 10 == 0:
            print("proj norms", proj_embs.norm(dim=1).mean().item())
            print("offsets norms", self.embedding_space.offsets.norm(dim=1).mean().item())
            print("normals norms", self.embedding_space.normals.norm(dim=1).mean().item())
            print("----------------------------------------\n\n")
            
            # print(self.)
        
        if self.steps % 50 == 0:
            import pdb
            pdb.set_trace()
        
        if self.computing_metrics or self.train_metrics:
            joints = self.embedding_space.get_joints(self.cprobs)
            preds = self.embedding_space.decide(joints)
            self.iou_fn.forward(preds, self.labels)
            self.acc_fn.forward(preds, self.labels)
            
        
    def update_model(self):
        
        valid_mask = self.labels <= self.tree.M - 1
        
        hce_loss = loss.CCE(
                    torch.moveaxis(self.cprobs, 1, -1)[valid_mask],
                    self.labels[valid_mask],
                    self.tree)
        
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
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                
                self.optimizer.zero_grad()
                
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
                self.compute_metrics()
                
            # compute and print the metrics for the validation data
            if self.val_metrics:
                self.compute_metrics_dataset(val_loader)
                
            
    def compute_metrics(self):
        accuracy = self.acc_fn.compute()
        miou = self.iou_fn.compute()
        print('     [train metrics]',
              '\n\n[accuracy per class]', accuracy.tolist(),
              '\n\n[miou per class]', miou.tolist(),
              '\n\n[mean accuracy]', accuracy.mean().item(),
              '\n\n[mean miou]', miou.mean().item(), '\n\n')
        self.iou_fn.reset()
        self.acc_fn.reset()
    
    
    def compute_metrics_dataset(self, loader: torch.utils.data.DataLoader):
        
        with torch.no_grad():
            metrics = {} 
            self.computing_metrics = True
            
            for images, labels, _ in loader:
                self.images = images.to(self.device)
                self.labels = labels.to(self.device).squeeze()
                self.data_forward()
                
            miou_per_class = self.iou_fn.compute()
            metrics['miou per class'] = torch.round(miou_per_class, decimals=5)
            metrics['miou'] = round(miou_per_class.mean().item(), 5)

            acc_per_class = self.acc_fn.compute()
            metrics['acc per class'] = torch.round(acc_per_class, decimals=5)
            metrics['acc'] = round(acc_per_class.mean().item(), 5)
            
            print('     [validation metrics]',
                '\n\n[accuracy per class]', metrics['acc per class'].tolist(),
                '\n\n[miou per class]', metrics['miou per class'].tolist(),
                '\n\n[mean accuracy]', metrics['acc'],
                '\n\n[mean miou]', metrics['miou'], '\n\n')
            
            self.iou_fn.reset()
            self.acc_fn.reset()
            self.computing_metrics = False
