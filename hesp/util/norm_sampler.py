import torch

class NormSampler:
    """
    Keep track of the mean L2 norms for collections of pixel embeddings belonging 
    to a (sample, class) combination and computes and stores the probability of
    each sample based on these norms.
    """
    
    def __init__(
        self,
        num_samples,
        num_classes,
        device
    ):
        self.norms = torch.zeros(num_samples, num_classes, device=device)
        self.sample_probs = torch.ones(num_samples, ) / num_samples
        
        
    def update_norms(
        self,
        embeddings,
        valid_labels,
        valid_mask,
        batch_indices,
    ):
        """ Update the L2 norms for the set of { (sample, class, index) } tuples in a batch. """
        with torch.no_grad():
            norms = torch.linalg.vector_norm(embeddings.moveaxis(1, -1)[valid_mask], dim=-1)
            slices = valid_mask.sum(dim=(1,2))
            bottom = 0 
            for i, s in enumerate(slices):
                top = bottom + s
                
                # skip zero size slices
                if top == bottom:
                    continue
                    
                sample_index = batch_indices[i]
                sample_labels = valid_labels[bottom : top]
                sample_classes = torch.unique(sample_labels)
                
                for c in sample_classes:
                    class_locations = sample_labels == c
                    self.norms[sample_index, c] = norms[bottom : top][class_locations].mean()
                
                bottom = top
                
        return None
            
                
    def compute_mean_sample_norms(self):
        """ 
        Compute the mean norms for every sample over all classes.
        Zeros due to not yet seen are interpolated with the mean.
        """
        means = torch.nan_to_num(self.norms.sum(dim=1) / (self.norms > 0.0).sum(dim=1))
        means[means == 0.0] = means[means != 0.0].mean()
        return means
    

    def compute_mean_class_norms(self):
        """ 
        Compute the mean norms for every class over all samples.
        Zeros due to not yet seen are interpolated with the mean.
        """
        means = torch.nan_to_num(self.norms.sum(dim=0) / (self.norms > 0.0).sum(dim=0))
        means[means == 0.0] = means[means != 0.0].mean()
        return means
    
                
    def compute_sample_probs(self):
        """
        Compute the sample probabilities as a softmax over the inverse of their mean embedding norms.
        Unseen samples get their norms interpolated by the mean of all others.
        """
        mean_norms = self.compute_mean_sample_norms()
        self.sample_probs = torch.softmax(1.0 / torch.log(mean_norms), dim=0)
        return None