""" Run this script before training to compute the means of Pascal used for normalization"""
import torch
import data_helpers

files, _, _ = data_helpers.make_data_splits(
    'pascal', limit=-1, split=(0.9, 0.1), seed=0, shuffle=True)
data = data_helpers.PascalDataset(files)
means = data_helpers.compute_dataset_means(data)
torch.save(means, 'datasets/pascal/means.pt')