import torch
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import EPS

EPS = torch.tensor(EPS, dtype=torch.float32)

def CCE(cond_probs: torch.Tensor, labels: torch.Tensor, tree: Tree) -> torch.Tensor:
        
    log_probs = torch.log(torch.maximum(cond_probs, EPS)) 
     
    log_sum_p = log_probs @ tree.hmat.T.to(cond_probs.device) 
    
    pos_logp = torch.gather(input=log_sum_p, index=labels[:, None], dim=1)

    return -pos_logp.mean()
