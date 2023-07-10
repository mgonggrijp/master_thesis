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
                sets values in self to average of the accumulated norms
                
            3. reset: 
                set all norm counts and values to zero
        """
        self.registry = torch.zeros(nsamples, nclasses, device=device)
        self.counter = torch.zeros(self.registry.size(), dtype=int, device=device)
        self.averaged = False
        self.nsamples = nsamples
        self.nclasses = nclasses
        self.device = device


    def update(
        self,
        valid_mask: torch.Tensor,
        valid_labels: torch.Tensor,
        proj_embs: torch.Tensor,
        batch_indeces: torch.Tensor,
    ):
        """
    Given the set of labels and projected embeddings from a batch, update
        the norm registry for the corresponding classes.
        """
        dev = valid_mask.device

        # compute the norms for valid pixels
        valid_embs = proj_embs.moveaxis(1, -1)[valid_mask]
        valid_norms = torch.linalg.vector_norm(valid_embs, dim=-1) 

        # get the slices that correspond to sets of pixels for specific samples after masking
        slices = valid_mask.sum(dim=(1,2))

        # duplicate the batch indeces such that they correspond to the valid labels one to one
        repeat_bidxs = batch_indeces.to(dev).repeat_interleave(slices, dim=0)

        # update the registry with the norms from the batch
        self.registry.index_put_((repeat_bidxs, valid_labels), valid_norms, accumulate=True)

        # update the counter to track for each (sample, class) combo how often it has been seen
        encountered = torch.ones(repeat_bidxs.size(0), dtype=int, device=dev)
        self.counter.index_put_((repeat_bidxs, valid_labels), encountered, accumulate=True)
        
        return None


    def average(self):
        """ 
        Compute the average of the norms for each (sample, class) combo
        given how many of that class are in each sample.
        """
        # safety to make sure to avoid division by zero
        if not self.averaged:
            non_zero = self.counter > 0
            self.registry[non_zero] /= self.counter[non_zero]
            self.averaged = True
            
        return None
    

    def reset(self):
        """ Reset the registry and counter to zero. """
        self.registry = torch.zeros_likes(self.registry)
        self.counter = torch.zeros_likes(self.counter)
        self.averaged = False
        
    
    def average_per_class(self):
        """ Compute and return the average norm per class over all samples. """
        
        # first average over the class counts per sample if not yet done so
        if not self.averaged:
            non_zero = self.counter > 0
            self.registry[non_zero] /= self.counter[non_zero]
            self.averaged = True
            
        # then average over the samples
        return self.registry.mean(dim=0)
        