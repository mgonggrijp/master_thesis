import torch
from hesp.hierarchy.tree import Tree

def compute_uncertainty_weights(embeddings: torch.Tensor, valid_mask: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """ 
    Given a set of embeddings of shape (batch, dim, height, width) 
    and a valid value mask of shape (batch, heigth width) compute the 
    uncertainty weights for each pixel embedding as 
                        uw = 1.0 / log(t + d)
    Where d is the set of hyperbolic distances of each pixel embedding 
    normalized by the maximum of these distance for the same image.
    """
    with torch.no_grad():
        # mask out the embeddings corresponding to ignore labels
        masked_embs = embeddings.moveaxis(1, -1)[valid_mask]
        
        # figure out how many elements get masked out per sample
        slices = valid_mask.sum(dim=(1,2))
        
        # each remaining pixel embedding gets one weight value
        weights = torch.ones(masked_embs.size(0), device=embeddings.device)
        
        sqrt_c = torch.sqrt(c)
        
        start = 0
        # go through samples
        for s in slices:
            # set slice end range
            end = start  + s
            
            if end != start:
                # take a slice of the masked embeddings for one sample and compute the hyperbolic distance to zero
                dist_c_zeros = 2.0 / sqrt_c * torch.arctanh(sqrt_c * torch.linalg.vector_norm(masked_embs[start:end], dim=-1))
                
                # for each sample, compute it's maximum distance
                sample_max = torch.amax(dist_c_zeros, dim=0)
                
                # then normalize the sample distances by the max
                max_normalized = dist_c_zeros / sample_max
                
                # compute the weights for this sample
                weights[start : end] = 1.0 / torch.log(1.02 + max_normalized) + 1.0
            
            # move to next slice
            start = end
            
    return weights


def compute_uweights_simple(embeddings, valid_mask):
    
    with torch.no_grad():
        # mask out the embeddings corresponding to ignore labels
        masked_embs = embeddings.moveaxis(1, -1)[valid_mask]
        
        # figure out how many elements get masked out per sample
        slices = valid_mask.sum(dim=(1,2))
        
        # each remaining pixel embedding gets one weight value
        weights = torch.ones(masked_embs.size(0), device=embeddings.device)
        
        start = 0
        # go through samples
        for s in slices:
            # set slice end range
            end = start  + s
            
            # compute L2 norms for a sample
            norms = torch.linalg.norm(masked_embs[start : end], dim=-1)
            
            # weights are inverse of the L2 norms
            weights[start : end] = 1.0 / norms
            
            # move to next slice
            start = end
            
    return weights    

def CCE(
    cprobs: torch.Tensor,
    labels: torch.Tensor,
    tree: Tree,
    uncertainty_weights: torch.Tensor = None
    ) -> torch.Tensor:
    """ 
    Compute the hierarchical cross-entropy loss as the -mean(correct(log prob)).
    When hierarchy is flat, reverts to cross-entropy. 
    """
    # numerical safety
    EPS = torch.tensor(1e-15)
    
    # get the hierarchy matrix that defines parent-child relationship
    hmat = tree.hmat.to(cprobs.device)
    
    # compute the log probs
    log_probs = torch.log(torch.maximum(cprobs, EPS)) 
    
    # multiply with hmat to get sums over chains of child-parents to get leaf probs
    log_sum_p = log_probs @ hmat.T 
    
    # collect the probabilities that correspond to the labels
    pos_logp = torch.gather(log_sum_p, 1, labels[:, None]).squeeze()

    # reweight the loss with uncertainty values if provided
    if isinstance(uncertainty_weights, torch.Tensor):
        weighted_pos_logp = pos_logp * uncertainty_weights
        return -weighted_pos_logp.mean()
    
    return -pos_logp.mean()
