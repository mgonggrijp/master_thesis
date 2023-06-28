import logging
import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from hesp.config.config import Config
from hesp.hierarchy.tree import Tree
import geoopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractEmbeddingSpace(ABC):
    """General purpose base class for implementing embedding spaces.
    Softmax function turns into hierarchical softmax only when the 'tree' has an hierarchal structure defined."""
    offsets = None
    normals = None
    curvature = None

    def __init__(self, tree: Tree, device, config: Config, train: bool = True, prototype_path: str = ''):
        self.tree = tree
        self.config = config
        self.dim = config.embedding_space._DIM
        self.device = device
        
        self.normals = torch.normal(
            mean=0.0,
            std=torch.full(
                size=[self.tree.M, self.dim],
                fill_value=0.05)).to(device)
        
        self.normals.requires_grad_()

        self.offsets = torch.zeros(
            size=[self.tree.M, self.dim],
            requires_grad=True,
            device=device)
        
        if config.embedding_space._GEOMETRY == 'hyperbolic':
            pcb = geoopt.manifolds.PoincareBall(
                c=config.embedding_space._INIT_CURVATURE)
            
            self.offsets = geoopt.ManifoldTensor(
                            self.offsets,
                            manifold=pcb,
                            device=device,
                            requires_grad=True)
            
            print('Offset manifold: ', self.offsets.manifold)


    def softmax(self, logits):

        # subtract the max value for numerical safety
        safe_logits = logits - torch.amax(logits, 1, keepdim=True) # sh (batch, num_nodes, height, width)
     
        exp_logits = torch.exp(safe_logits)

        # matmul with sibmat to compute the sums over siblings for the normalizer Z
        Z = torch.tensordot(
            exp_logits,
            self.tree.sibmat.to(self.device),
            dims=[[1], [-1]])  # sh (batch, height, width, num_nodes)
        
        # reshape back to (batch, num_nodes, height, width)
        Z_reshaped = torch.moveaxis(Z, -1, 1)  
        
        # compute conditional probabilities with numerical safety
        cond_probs = exp_logits / (Z_reshaped + 1e-15)
        
        return cond_probs 

    def decide(self, probs: torch.Tensor, unseen: list = []):
        """ Decide on leaf class from probabilities. """
        with torch.no_grad():
            cls_probs = probs[:, :self.tree.K, :]
            hmat = self.tree.hmat[:self.tree.K, :self.tree.K].to(probs.device)

            cls_probs = torch.tensordot(cls_probs, hmat, dims=[[1], [-1]])
            cls_probs = torch.moveaxis(cls_probs, [1, 3], [2, 1])  # sh (batch, K, height, width)
            
            if len(unseen):
                cls_gather = cls_probs[:, unseen, :]
                predict_ = torch.argmax(cls_gather, dim=1)
                predictions = torch.tensor(unseen)[predict_]
            else:
                predictions = torch.argmax(cls_probs, dim=1)

        return predictions

    def run(self, embeddings, offsets=None, normals=None, curvature=None):
        """ Calculates (joint) probabilities for incoming embeddings. Assumes embeddings are already on manifold. """
        if offsets is None:
            offsets = self.offsets
        if normals is None:
            normals = self.normals
        if curvature is None:
            curvature = self.curvature

        logits = self.logits(
            embeddings=embeddings,
            offsets=offsets,
            normals=normals,
            curvature=curvature)  # shape (B, M, H, W)

        cond_probs = self.softmax(logits)  # shape (B, M , H, W)
        joints = self.get_joints(cond_probs)

        return joints, cond_probs

    def get_joints(self, cond_probs):
        """ Calculates joint probabilities based on conditionals """
        log_probs = torch.log(cond_probs + 1e-15) 
        
        hmat = self.tree.hmat.to(cond_probs.device)

        log_sum_p = torch.tensordot(log_probs, hmat, dims=[[1], [-1]])  # sh (B, H, W, M)

        joints = torch.exp(log_sum_p)

        # reshape back to (B, M, H, W)
        final_joints = torch.moveaxis(joints, [1, 3], [2, 1])

        return final_joints

    @ abstractmethod
    def logits(self, embeddings, offsets=None, normals=None, curvature=None):
        """ Returns logits to pass to (hierarchical) softmax function."""
        pass
