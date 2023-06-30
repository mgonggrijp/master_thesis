import torch
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import EPS

EPS = torch.tensor(EPS, dtype=torch.float32)

def CCE(cond_probs: torch.Tensor, labels: torch.Tensor, tree: Tree, steps) -> torch.Tensor:
        
    hmat = tree.hmat.to(cond_probs.device)
     
    log_probs = torch.log(
        torch.maximum(cond_probs, EPS)) 
     
    log_sum_p = log_probs @ hmat.T 

    # if steps % 10 == 0:
    #     _, indeces = log_sum_p[:, :21].max(-1)
    #     bincounts = torch.bincount(indeces.flatten(), minlength=21)

    #     for i in range(21):
    #         if bincounts[i] != 0:
    #             print('Count for class {}:'.format(i), bincounts[i])
                
    #     print('\n\n------------------------------------------------')
    
    pos_logp = torch.gather(
        input=log_sum_p, index=labels[:, None], dim=1)

    return -pos_logp.mean()
