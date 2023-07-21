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
                    
                sample_index = indices[i]
                sample_labels = valid_labels[bottom : top]
                sample_classes = torch.unique(sample_labels)
                
                for c in sample_classes:
                    class_locations = sample_labels == c
                    self.norms[sample_index, c] = norms[bottom : top][class_locations].mean()
                
                bottom = top
                
    def compute_sample_probs(self):
        mean_sample_norms = self.norms.mean(dim=1)
        mean_sample_norms[mean_sample_norms == 0.0] = mean_sample_norms[mean_sample_norms != 0.0].mean()
        self.sample_probs = torch.softmax(1.0 / mean_sample_norms, dim=0)