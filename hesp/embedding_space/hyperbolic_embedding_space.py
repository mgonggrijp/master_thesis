import os

import numpy as np
import torch


from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import torch_exp_map_zero
from hesp.util.layers import torch_hyp_mlr


class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):

    def __init__(self, tree: Tree, config: Config, device: torch.device, train: bool = True, prototype_path: str = ''):
        super().__init__(tree=tree, config=config, device=device, train=train, prototype_path=prototype_path)
        
        self.geometry = 'hyperbolic'
        self.device = device
        # sanity check
        assert self.geometry == config.embedding_space._GEOMETRY, 'config geometry does not match embedding spaces'

        curv_init = self.config.embedding_space._INIT_CURVATURE

        if not train:
            self.c_npy = np.load(os.path.join(prototype_path, 'c.npy'))
            curv_init = self.c_npy

        self.curvature = torch.tensor(
            curv_init,
            requires_grad=False,
            dtype=torch.get_default_dtype()).to(device)
        

    def project(self, embeddings):
        return torch_exp_map_zero(embeddings, c=self.curvature)
    

    def logits(self, embeddings, offsets, normals, curvature):
        return torch_hyp_mlr(inputs=embeddings, c=curvature, P_mlr=offsets, A_mlr=normals)
