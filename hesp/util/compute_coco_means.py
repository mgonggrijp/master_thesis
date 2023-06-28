""" Run this script before training to compute the means of COCO used for normalization"""
import torch
import data_helpers 

limit = int(1.5e4) # change depending on memory
shuffle = False
coco = data_helpers.CocoDataset(
        scale=(None, None),
        output_size=(250, 250),
        limit=limit,
        shuffle=shuffle,
        mode='train',
        transforms=data_helpers.resize_transforms, # needed to ensure output sizes are uniform
        use_transforms=True,)
means = data_helpers.compute_dataset_means(coco)
torch.save(means, 'datasets/coco/means.pt')
