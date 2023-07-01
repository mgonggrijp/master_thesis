import torch
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import EPS

EPS = torch.tensor(EPS)

def CCE(
    cond_probs: torch.Tensor,
    labels: torch.Tensor,
    tree: Tree,
    class_weights: torch.Tensor) -> torch.Tensor:
    
    hmat = tree.hmat.to(cond_probs.device)
     
    log_probs = torch.log(
        torch.maximum(cond_probs, EPS)) 
     
    log_sum_p = log_probs @ hmat.T 
    
    pos_logp = torch.gather(
        input=log_sum_p, index=labels[:, None], dim=1).squeeze()

    weighted_pos_logp = pos_logp * class_weights[labels]    

    return -weighted_pos_logp.mean()
