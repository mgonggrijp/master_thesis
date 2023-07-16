import torch

class NormRegistry:
    
    def __init__(
        self,
        nsamples: int,
        nclasses: int,
        device = 'cuda:0'
    ):
        """
        Registry used to store embedding norms for each (sample_i, class_k) combination.
        It's functions are:
        
            1. update: 
                given a batch of embeddings and labels, adds the corresponding norms
                to the location for the (sample_idx, class_idx) in the registry
                
            2. average: 
                updates the average registry with the norms accumulated so far
                
            3. reset: 
                set the accumulation registry, mean registry and counter to zero
                
            4. average_per_class:
                compute the average norm per class over all samples and pixels
                
            5. average_per_sample:
                compute the average norm per sample over all classes and pixels
        """
        self.accumulate_registry = torch.zeros(nsamples, nclasses, device=device)
        self.mean_registry = torch.zeros_like(self.accumulate_registry)
        self.counter = torch.zeros(self.accumulate_registry.size(), dtype=int, device=device)
        
        self.averaged = False
        self.nsamples = nsamples
        self.nclasses = nclasses
        self.device = device


    def update(
        self,
        valid_mask: torch.Tensor,
        valid_labels: torch.Tensor,
        embeddings: torch.Tensor,
        batch_indeces: torch.Tensor,
    ): 
        """
        Update the accumulation registry with norms for a batch of embeddings.
        """
        
        with torch.no_grad():
            self.averaged = False
            
            dev = valid_mask.device

            # compute the norms for valid pixels
            valid_embs = embeddings.moveaxis(1, -1)[valid_mask]
            
            valid_norms = torch.linalg.vector_norm(valid_embs, dim=-1) 

            # get the slices that correspond to sets of pixels for specific samples after masking
            slices = valid_mask.sum(dim=(1,2))

            # duplicate the batch indeces such that they correspond to the valid labels one to one
            repeat_bidxs = batch_indeces.to(dev).repeat_interleave(slices, dim=0)

            # update the registry with the norms from the batch
            self.accumulate_registry.index_put_((repeat_bidxs, valid_labels), valid_norms, accumulate=True)

            # update the counter to track for each (sample, class) combo how often it has been seen
            encountered = torch.ones(repeat_bidxs.size(0), dtype=int, device=dev)
            self.counter.index_put_((repeat_bidxs, valid_labels), encountered, accumulate=True)
            
        return None


    def average(self):
        """ 
        Computes the average of the accumulation registry so far and stores it in the mean registry.
        """
        # safety to make sure to avoid division by zero
        with torch.no_grad():
            if not self.averaged:
                    counter_non_zero = self.counter > 0

                    # compute the average over the values accumulated so far
                    self.mean_registry[counter_non_zero] = self.accumulate_registry[counter_non_zero] / self.counter[counter_non_zero]

                    # for avoiding averaging twice in a row to save computation
                    self.averaged = True
                
        return None
    

    def reset(self):
        self.accumulate_registry *= 0.0
        self.mean_registry *= 0.0
        self.counter *= 0
        self.averaged = False
        
    
    def average_per_class(self):
        """ Compute the average norm per class over pixels and samples. """
        # make sure that the mean registry is up-to-date with the recently collected norms
        if not self.averaged:
            self.average()
            
        # for each class compute the average over the samples; 
        with torch.no_grad():        
            return torch.nan_to_num(self.mean_registry.sum(dim=0) / (self.mean_registry > 0).sum(dim=0))
        
    
    def average_per_sample(self):
        """ Compute the average norm per sample over pixels and classes. """
        # update the mean registry with the averages per (sample, class) over pixels.
        if not self.averaged:
            self.average()

        # for each sample compute the average over the classes;      
        with torch.no_grad():
            return torch.nan_to_num(self.mean_registry.sum(dim=1) / (self.mean_registry > 0).sum(dim=1))
            