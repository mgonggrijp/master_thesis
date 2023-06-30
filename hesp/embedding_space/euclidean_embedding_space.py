
from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.layers import torch_euc_mlr
from hesp.config.config import Config
from hesp.hierarchy.tree import Tree
import torch


class EuclideanEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, tree: Tree, config: Config,):
        super().__init__(tree, config)
        
        self.geometry = 'euclidean'
        # sanity check
        assert self.geometry == config.embedding_space._GEOMETRY, 'config geometry does not match embedding spaces'

    def project(self, embeddings):
        return embeddings

    def logits(self, embeddings: torch.tensor, offsets: torch.tensor, normals: torch.tensor):
        return torch_euc_mlr(embeddings, P_mlr=offsets, A_mlr=normals)
