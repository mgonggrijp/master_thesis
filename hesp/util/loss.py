import torch
from torch.linalg import vector_norm as vnorm
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


def compute_class_norm_weights(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_labels: torch.Tensor,
    method: str = 'invert'
    ):
    
    """ Compute the pixel weights based on the norms within each class and sample. """
    
    with torch.no_grad():
        # slices that correspond to samples after masking
        sample_slices = valid_mask.sum(dim=(1,2))
        
        # compute the L2 norms of the norms corresponding to valid pixels
        valid_norms = vnorm( embeddings.moveaxis(1, -1)[valid_mask], dim=-1 )
        
        # each pixel after applying the valid mask gets one weight
        weights = torch.zeros_like(valid_norms)
        
        # go through the slices that correspond to samples after masking 
        start = 0
        for s in sample_slices:

            end = start + s

            # avoid making empty slices, rare case if all labels are 'ignore' 
            if end == start:
                start = end
                continue

            # get the norms and labels corresponding to the current sample 
            sample_norms = valid_norms[start : end]
            sample_labels = valid_labels[start : end]

            # get the set of classes present in the current sample slice
            class_set = torch.unique(sample_labels)

            for c in class_set:
                # find the locations of the current class in the current sample
                class_locations = sample_labels == c
                
                # compute the norms per class for current sample
                class_norms = sample_norms[class_locations]
            
                # find the maximum norm for the (sample, class)
                max_class_norm = torch.amax(class_norms, dim=0)

                # normalize the norms for the same (sample, class) by their maximum
                normalized_norms = class_norms / max_class_norm

                # compute and store the weights based on these norms with the selected scheme
                if method == 'invert':
                    weights[start : end][class_locations] = 1.0 / normalized_norms

            # move to next sample slice
            start = end

    return weights


def compute_weights(**weight_args):
    
    method = weight_args['method']
    embeddings = weight_args['embeddings']
    valid_mask = weight_args['valid_mask']
    valid_labels = weight_args['valid_labels']
    
    if method == 'uncertainty':
        return compute_uncertainty_weights(embeddings, valid_mask)
    
    elif method == 'class_based':
        return compute_class_norm_weights(embeddings, valid_mask, valid_labels)
    
    else:
        raise NotImplementedError()
    

def CCE(
    cprobs: torch.Tensor,
    labels: torch.Tensor,
    tree: Tree,
    weights: torch.Tensor = None
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
    if isinstance(weights, torch.Tensor):
        weighted_pos_logp = pos_logp * weights
        return -weighted_pos_logp.mean()
    
    return -pos_logp.mean()
