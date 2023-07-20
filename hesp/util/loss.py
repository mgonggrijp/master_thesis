import torch
from hesp.hierarchy.tree import Tree


def compute_uncertainty(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor
    ) -> torch.Tensor:
    """ 
    Compute the inverse L2 norm for a collection of valid pixel embeddings normalized
    by the maximum norm for the same image.
    """
    
    # compute the slices that correspond to samples after masking
    slices = valid_mask.sum(dim=(1,2))
    
    # mask the invalid pixels and compute the norms of their embeddings
    valid_norms = torch.linalg.vector_norm(embeddings.moveaxis(1, -1)[valid_mask], dim=-1)

    # each pixel gets one weight value    
    weights = torch.ones_like(valid_norms)
    
    lower = 0
    
    for s in slices:
        upper = lower + s 

        if upper != lower:
            # get the slice of norms corresponding to the current sample        
            sample_norms = valid_norms[lower : upper]
            
            # compute the maximum norm for this sample
            max_sample_norm = sample_norms.amax(dim=0)
            
            # normalize by the maximum and take it's inverse; smaller norms --> larger weight
            weights[lower : upper] = max_sample_norm / sample_norms
        
        lower = upper
    
    return weights    


def compute_class_uncertainty(
    embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_labels: torch.Tensor,
    ):
    
    """ Compute the pixel weights based on the norms within each class and sample. """
    
    # slices that correspond to samples after masking
    sample_slices = valid_mask.sum(dim=(1,2))
    
    # compute the L2 norms of the norms corresponding to valid pixels
    valid_norms = torch.linalg.vector_norm( embeddings.moveaxis(1, -1)[valid_mask], dim=-1 )
    
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
            
            # retrieve the norms per class for current sample
            class_norms = sample_norms[class_locations]
        
            # find the maximum norm for the (sample, class)
            max_class_norm = torch.amax(class_norms, dim=0)

            # normalize by the maximum and take it's inverse; smaller norms --> larger weight
            weights[start : end][class_locations] = max_class_norm / class_norms

        # move to next sample slice
        start = end

    return weights


def compute_uncertainty_loss_augmentation(**weight_args):
    
    method = weight_args['method']
    embeddings = weight_args['embeddings']
    valid_mask = weight_args['valid_mask']
    valid_labels = weight_args['valid_labels']
    cprobs = weight_args['cond_probs']
    
    if method == 'basic_weights':
        return compute_uncertainty(embeddings, valid_mask)
    
    elif method == 'class_weights':
        return compute_class_uncertainty(embeddings, valid_mask, valid_labels)
    
    elif method == 'encourage':
        return compute_encouragement_loss(embeddings, cprobs, valid_labels, valid_mask)    

    else:
        raise NotImplementedError()
    
    
def compute_encouragement_loss(
    embs: torch.Tensor,
    probs: torch.Tensor,
    valid_labels: torch.Tensor,
    valid_mask: torch.Tensor
    ) -> torch.Tensor:
    """ 
    Computes the norm term loss which punishes large embedding norms
    when the most likely class is incorrect and punishes small norms
    vice versa.
    """
    
    # mask the probabilities and embeddings to remove ignored pixels
    valid_probs = probs.moveaxis(1, -1)[valid_mask]
    valid_embs = embs.moveaxis(1, -1)[valid_mask]
    
    # get the locations of the pixels for which the maximum probability is the correct class
    correct_preds = torch.argmax(valid_probs, dim=-1) == valid_labels

    # compute the L2 norms and their inverse for the embeddings
    norms = torch.linalg.vector_norm(valid_embs, dim=-1)
    inverse_norms = 1.0 / norms[correct_preds]

    # correctly predicted pixels get punished for small norms and incorrect pixels get punished for large norms
    return ( inverse_norms.mean() + norms[~correct_preds].mean() ) / 2.0


def CCE(
    cprobs: torch.Tensor,
    labels: torch.Tensor,
    tree: Tree,
    **kwargs,
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
    
    # check kwargs for uncertainty tensors used for loss augmentation
    kwarg_keys = kwargs.keys()
    
    # weight loss with class based uncertainty weights
    if 'class_weights' in kwarg_keys:
        return -(pos_logp * kwargs['class_weights']).mean()

    # weight loss with basic uncertainty weights        
    elif 'basic_weights' in kwarg_keys:
        return -(pos_logp * kwargs['basic_weights']).mean()
    
    # add the uncertainty based encouragement term to the loss
    elif 'encourage' in kwarg_keys:
        return -pos_logp.mean() + kwargs['encourage']
        
    # return default loss    
    return -pos_logp.mean()
