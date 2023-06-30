import os
import numpy as np
import torch
from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import torch_exp_map_zero
from hesp.util.layers import torch_hyp_mlr


class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, tree: Tree, config: Config):
        super().__init__(tree, config)
        
        self.geometry = 'hyperbolic'

        curv_init = self.config.embedding_space._INIT_CURVATURE

        self.curvature = torch.nn.Parameter(
            torch.tensor(curv_init),
            requires_grad=False)

    def project(self, embeddings):
        return torch_exp_map_zero(embeddings, self.curvature, self.EPS)
    

    def logits(self, embeddings, offsets, normals, curvature):
        return torch_hyp_mlr(
            inputs=embeddings, c=curvature, P_mlr=offsets, A_mlr=normals, EPS=self.EPS)
