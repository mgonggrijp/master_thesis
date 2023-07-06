import torch
from hesp.hierarchy.tree import Tree
from torch.linalg import vector_norm as vnorm
from torch import sqrt, log, maximum, gather, zeros

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
        batch_size = embeddings.size(0)
        
        # mask out the embeddings corresponding to ignore labels
        masked_embs = embeddings.moveaxis(1, -1)[valid_mask]
        
        # figure out how many elements get masked out per sample
        post_mask_indeces = valid_mask.sum(dim=(1,2))
        
        # each remaining pixel embedding gets one weight value
        weights = zeros(masked_embs.size(0))
        
        start = 0
        sqrt_c = sqrt(c)
        
        # go through samples
        for i in range(batch_size):
            # set slice end range
            end = start + post_mask_indeces[i]
            
            # take a slice of the masked embeddings for one sample
            sample_slice = masked_embs[start:end]
            
            # compute the hyperbolic zero distances
            dist_c_zeros = 2.0 / sqrt_c * (sqrt_c * vnorm(sample_slice, dim=-1))
            
            # for each sample, compute it's maximum distance
            sample_max = torch.amax(dist_c_zeros, dim=0)
            
            # then normalize the sample distances by the max
            max_normalized = dist_c_zeros / sample_max
            
            # compute the weights for this sample
            weights[start : end] = 1.0 / log(1.02 + max_normalized) + 1.0
            
            # move to next slice
            start = end
            
    return weights


def CCE(cprobs: torch.Tensor, labels: torch.Tensor, tree: Tree, class_weights: torch.Tensor) -> torch.Tensor:
    """ 
    Compute the hierarchical cross-entropy loss as the -mean(correct(log prob)).
    When hierarchy is flat, reverts to cross-entropy. 
    """
    # numerical safety
    EPS = torch.tensor(1e-15)
    
    # get the hierarchy matrix that defines parent-child relationship
    hmat = tree.hmat.to(cprobs.device)
    
    # compute the log probs
    log_probs = log(maximum(cprobs, EPS)) 
    
    # multiply with hmat to get sums over chains of child-parents to get leaf probs
    log_sum_p = log_probs @ hmat.T 
    
    # collect the probabilities that correspond to the labels
    pos_logp = gather(log_sum_p, 1, labels[:, None]).squeeze()
    
    # reweight the log probs with the class weights to balance class frequency differences 
    # class_weighted_pos_logp = pos_logp # * class_weights[labels] * 10.0  # rescale to adjust for class weights
    
    return -pos_logp.mean()
