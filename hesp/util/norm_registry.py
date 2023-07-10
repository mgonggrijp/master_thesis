import torch

class NormRegistry:
    
    def __init__(
        self,
        nsamples: int,
        nclasses: int,
        device = 'cuda:0'
    ):
        """
        Registry used to store embedding norms for each (sample, class) combination.
        It has three main functionalities:
        
            1. update: 
                given a batch of embeddings and labels, adds the corresponding norms
                to the location for the (sample_idx, class_idx) in the registry
                
            2. average: 
                updates the average registry with the norms accumulated so far
                
            3. reset: 
                set the accumulation registry, mean registry and counter to zero
                
            4. average_per_class:
                compute the average norm per class over all samples
        """
        self.accumulate_registry = torch.zeros(nsamples, nclasses, device=device)
        # for storing intermediate averages
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
        Compute the average of the values seen so far. When an average (sample_i, class_k) 
        value already exists, the corresponding average is computed as the average 
        of the current mean and the average of the newly accumulated values.
        Assumes embedding norms are practically never zero.
        """
        # safety to make sure to avoid division by zero
        with torch.no_grad():
            if not self.averaged:
                
                    counter_non_zero = self.counter > 0

                    # check if the average registry already has values
                    averages_zero = self.mean_registry == 0

                    # new (sample_i, class_k) norms are those which have been counted and not yet averaged over
                    new_values = counter_non_zero & averages_zero
                    
                    # old (sample_i, class_k) norms that have already accumulated at least one value
                    old_values = counter_non_zero & ~averages_zero

                    # compute and store the average untill now for the averages that have not been seen yet
                    self.mean_registry[new_values] = self.accumulate_registry[new_values] / self.counter[new_values]

                    # for the values that have been averaged are replaced with the average so far and the average of the new values
                    self.mean_registry[old_values] =  (self.mean_registry[old_values] + \
                                                       self.accumulate_registry[old_values] / self.counter[old_values]) / 2.0

                    # reset the counter the continue for the next sample stream
                    self.counter = torch.zeros_like(self.counter)
                    
                    # reset the accumulation registry
                    self.accumulate_registry *= 0.0

                    # s.t. you can't average twice in a row without updating first
                    self.averaged = True
                
        return None
    

    def reset(self):
        """ Reset the accumulation registry, average registry and the counter. """
        self.accumulate_registry *= 0.0
        self.mean_registry *= 0.0
        self.counter *= 0
        self.averaged = False
        
    
    def average_per_class(self):
        """ Compute and return the average norm per class over all samples. """
        # make sure that the mean registry is up-to-date with the recently collected norms
        if not self.averaged:
            self.average()
            
        with torch.no_grad():        
        # then average over the samples
            return self.mean_registry.mean(dim=0)
        