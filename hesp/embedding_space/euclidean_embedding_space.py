
from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.layers import torch_euc_mlr
from hesp.config.config import Config
from hesp.hierarchy.tree import Tree
import torch

torch.set_default_dtype(torch.float32)

class EuclideanEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, tree: Tree, device, config: Config, train: bool = True, prototype_path: str = ''):
        super().__init__(tree=tree, device=device, config=config,
                         train=train, prototype_path=prototype_path)
        self.geometry = 'euclidean'
        # sanity check
        assert self.geometry == config.embedding_space._GEOMETRY, 'config geometry does not match embedding spaces'


    def project(self, embeddings, curvature=0):
        return embeddings
    

    def logits(self, embeddings: torch.tensor, offsets: torch.tensor, normals: torch.tensor, curvature=0):
        return torch_euc_mlr(embeddings, P_mlr=offsets, A_mlr=normals)
