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


class AbstractEmbeddingSpace(torch.nn.Module):
    def __init__(self, tree: Tree, config: Config):
        super().__init__()

        self.tree = tree
        self.config = config
        self.dim = config.embedding_space._DIM
        self.EPS = torch.tensor(1e-15)
        self.PROJ_EPS = torch.tensor(1e-3)
        
        std_normals = torch.full(
            size=[self.tree.M, self.dim], fill_value=0.05)
        
        std_offsets = torch.full(
            size=[self.tree.M, self.dim], fill_value=0.001)
        
        self.normals = torch.nn.Parameter(
            torch.normal(0.0, std_normals),
            requires_grad=True)
        
        if config.embedding_space._GEOMETRY == 'hyperbolic':
            
            pcb = geoopt.manifolds.PoincareBall(
                c=config.embedding_space._INIT_CURVATURE)
            
            print('[curvature]', config.embedding_space._INIT_CURVATURE)
            
            self.offsets = geoopt.ManifoldParameter(
                            torch.normal(0.0, std_offsets),
                            manifold=pcb,
                            requires_grad=True)
            
        else:
            self.offsets = torch.nn.Parameter(
                torch.normal(0.0, std_offsets), requires_grad=True)
            

    def forward(self, x: torch.Tensor, steps):
        """ Given a set of vectors embedded in Euclidean space,
            compute the conditional probabilities for each of these. 
            
            args:
                x: sh (batch, dim, height, width) set of dim sized vectors
                
            returns
                cprobs: sh (batch, nclasses, height, width) conditional probability
                        for each class given each embedding. """
        
        # embed the vectors in poincareball
        x = self.project(x)
        
        if steps % 50 == 0:
            with torch.no_grad():
                print('[embedding norms]',
                    torch.linalg.vector_norm(x, dim=1).mean().item() )
        
        # compute the conditional probabilities
        x = self.run(x)
        
        return x
        

    def softmax(self, logits):

        # subtract the max value for numerical safety
        safe_logits = logits - torch.amax(logits, 1, keepdim=True) # sh (batch, num_nodes, height, width)
     
        exp_logits = torch.exp(safe_logits)

        # matmul with sibmat to compute the sums over siblings for the normalizer Z
        Z = torch.tensordot(
            exp_logits, self.tree.sibmat, dims=[[1], [-1]])  # sh (batch, height, width, num_nodes)
        
        # reshape back to (batch, num_nodes, height, width)
        Z_reshaped = torch.moveaxis(Z, -1, 1)  
        
        # compute conditional probabilities with numerical safety
        cond_probs = exp_logits / (Z_reshaped + 1e-15)
        
        return cond_probs 


    def decide(self, probs: torch.Tensor, unseen: list = []):
        """ Decide on leaf class from probabilities. """
        with torch.no_grad():
            cls_probs = probs[:, :self.tree.K, :]
            hmat = self.tree.hmat[:self.tree.K, :self.tree.K]

            cls_probs = torch.tensordot(cls_probs, hmat, dims=[[1], [-1]])
            cls_probs = torch.moveaxis(cls_probs, [1, 3], [2, 1])  # sh (batch, K, height, width)
            
            if len(unseen):
                cls_gather = cls_probs[:, unseen, :]
                predict_ = torch.argmax(cls_gather, dim=1)
                predictions = torch.tensor(unseen)[predict_]
            else:
                predictions = torch.argmax(cls_probs, dim=1)

        return predictions


    def run(self, embeddings):
        """ Calculates (joint) probabilities for incoming embeddings. Assumes embeddings are already on manifold. """

        logits = self.logits(
            embeddings=embeddings,
            offsets=self.offsets,
            normals=self.normals,
            curvature=self.curvature)  # shape (B, M, H, W)

        cond_probs = self.softmax(logits)  # shape (B, M , H, W)
        
        # joints = self.get_joints(cond_probs)

        return cond_probs


    def get_joints(self, cond_probs):
        """ Calculates joint probabilities based on conditionals """
        log_probs = torch.log(cond_probs + 1e-15) 
        
        hmat = self.tree.hmat

        log_sum_p = torch.tensordot(log_probs, hmat, dims=[[1], [-1]])  # sh (B, H, W, M)

        joints = torch.exp(log_sum_p)

        # reshape back to (B, M, H, W)
        final_joints = torch.moveaxis(joints, [1, 3], [2, 1])

        return final_joints


    @ abstractmethod
    def logits(self, embeddings, offsets=None, normals=None, curvature=None):
        """ Returns logits to pass to (hierarchical) softmax function."""
        pass
    
    
    @ abstractmethod
    def project(self, embeddings, curvature):
        pass
