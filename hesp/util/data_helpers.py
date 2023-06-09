"""Utility functions for preprocessing data and setting up dataloaders from jpgs / pngs"""
import torchvision
import torch
import random
import os
import PIL
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision
import torch
import random
import os
import PIL
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

ROOT = '/home/mgonggri/master_thesis/'


# change depending on where the folder is located
PASCAL_ROOT = ROOT + "datasets/pascal/data/"
COCO_ROOT = ROOT + "datasets/coco/data/"

def imshow(images, labels, save_folder):
    for i, (img, lab) in enumerate( zip(images, labels) ):
        img = img.moveaxis(0, -1).numpy()
        lab = lab.squeeze().numpy()
        
        plt.imshow(img)
        plt.savefig(save_folder + 'img_{}.png'.format(i))
        plt.show()
        plt.close()
        
        plt.imshow(lab)
        plt.savefig(save_folder + 'lab_{}.png'.format(i))
        plt.show()
        plt.close()


def transforms(dataset, image: torch.Tensor, labels: torch.Tensor):
        """ Applies transformations to the image during inference and training. 
        
        Training transformations are:
            Random resize, random crop and random horizontal flip.
            
        Inference transformations are:
            Center crop to (N, N) where N is the largest of the width and height,
            followed by a resize to output shape. 
            """
        
        if dataset.is_training:
            _, h, w = image.shape
            rescale_factor = min(dataset.scale) + random.random() * (max(dataset.scale) - min(dataset.scale))
            new_size = [int(rescale_factor * h), int(rescale_factor * w)]
            
            image = TF.resize(
                image, new_size,
                torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True)
            
            labels = TF.resize(
                labels[None, :],
                new_size,
                torchvision.transforms.InterpolationMode.NEAREST,
                antialias=False)
            
            _, h, w = image.shape
            top  = random.randint(0, int(0.5 * h)) # uniform offset
            left  = random.randint(0, int(0.5 * w)) # uniform offset
            
            image = TF.crop(
                image, top, left, int(0.5 * h), int(0.5 * w))
            
            labels = TF.crop(
                labels, top, left, int(0.5 * h), int(0.5 * w))
            
            if random.random() > .5:
                image = TF.hflip(image)
                labels = TF.hflip(labels)
                
            image = torchvision.transforms.functional.adjust_brightness(
                image,
                random.uniform(0.8, 1.2))

            image = torchvision.transforms.functional.adjust_contrast(
                image,
                random.uniform(0.8, 1.2))
        
        _, h, w = image.shape
        if h > w:
            crop_h, crop_w = h, h
            
        else:
            crop_h, crop_w = w, w
        
        new_labels = torch.ones(crop_h, crop_w, dtype=labels.dtype) * 255
        new_image = torch.zeros(image.size(0), crop_h, crop_w, dtype=image.dtype)
        
        new_labels[:h, :w] = labels
        new_image[:, :h, :w] = image
        
        image = TF.resize(
            new_image,dataset.output_size,torchvision.transforms.InterpolationMode.BILINEAR,antialias=True)
        
        labels = TF.resize(
            new_labels[None, :], dataset.output_size, torchvision.transforms.InterpolationMode.NEAREST, antialias=False)

        labels = labels.squeeze()
        
        image = image - dataset.means[:, None, None]

        return image, labels.unsqueeze(0)


def compute_dataset_means(dataset: torch.utils.data.Dataset):
    """ compute the per-channel means for a dataset """
    means = torch.zeros(3, )
    data_size = len(dataset)
    
    for image, _ in dataset:
        means += image.view(3, -1).mean(dim=-1) / data_size
        
    return means.squeeze()


class PascalDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            files: list,
            output_size: tuple,
            scale: tuple,
            pad_label = 255,
            transforms = transforms,
            is_training: bool = False,
            shuffle: bool = False,
            use_transforms: bool = False):
        
        convert_tensor = torchvision.transforms.ToTensor()
        self.data = []
        self.output_size = output_size
        self.is_training = is_training
        self.transforms = transforms
        self.scale = scale
        self.use_transforms = use_transforms
        self.pad_label = pad_label

        # fill in AFTER init
        self.means = None 

        if shuffle:
            random.shuffle(files)

        for index, file in enumerate(files):
            labels = PIL.Image.open(PASCAL_ROOT + "SegmentationClassAug/" + file + ".png", formats=["PNG"])
            labels = torch.tensor(np.asarray(labels)).to(torch.long)

            image = PIL.Image.open(PASCAL_ROOT + "JPEGImages/" + file + ".jpg", formats=["JPEG"])
            image = convert_tensor(image)
            
            self.data.append([image, labels, torch.tensor(index, dtype=torch.long,)])


    def __len__(self, ):
        return len(self.data)


    def __getitem__(self, idx):
        image, labels, index = self.data[idx]

        if self.use_transforms:
            image, labels = self.transforms(
                self, image=image, labels=labels.squeeze())

        return image, labels, index


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            output_size: tuple = [513, 513],
            scale: tuple = [0.5, 2.0],
            mode : str = 'val',
            pad_label = 255,
            limit: int = 10,
            transforms = transforms,
            is_training: bool = False,
            shuffle: bool = False,
            use_transforms: bool = False) -> torch.utils.data.Dataset:
        
        if mode not in ['train', 'val']:
            raise ValueError('Incorrect mode format, choose one of [train, val].')
        
        convert_tensor = torchvision.transforms.ToTensor()
        self.data = []
        self.output_size = output_size
        self.is_training = is_training
        self.transforms = transforms
        self.scale = scale
        self.use_transforms = use_transforms
        self.mode = mode
        self.pad_label = pad_label
        self.means = None # gets a value after initialization

        label_folder = COCO_ROOT + "annotations/" + mode + "2017/"
        label_files = os.listdir(label_folder)

        if shuffle:
            random.shuffle(label_files)

        for index, label_png in enumerate(label_files[:limit]):
            try:
                image_file = COCO_ROOT + mode + '2017/' + label_png.split('.')[0] + '.jpg' 
                image = convert_tensor(PIL.Image.open(image_file, formats=["JPEG"]))
            except:
                continue

            # skip the gray scale images
            if image.shape[0] == 1:
                continue

            labels = torch.tensor(
                np.asarray(PIL.Image.open(label_folder + label_png, formats=["PNG"])),
                dtype=torch.long)

            self.data.append((image, labels, torch.tensor(index, dtype=torch.long,)))

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        image, labels, index = self.data[idx]

        if self.use_transforms:
            image, labels = self.transforms(
                self,
                image=image,
                labels=labels.unsqueeze(0))

        return image, labels, index


def make_data_splits(data_name, limit, split, shuffle = False, seed = None):
    
    if type(seed) == float:
        random.seed(seed)
    
    """ Ignore the pre-determined train-val split and return a custom train-val
    file split based on the given split value. Used for Pascal only. """

    if data_name == 'pascal':
        files = [x.split('.')[0] for x in os.listdir(PASCAL_ROOT + "SegmentationClassAug")]
        
    if data_name == 'coco':
        files = [x.split('.')[0] for x in os.listdir(COCO_ROOT + "SegmentationClass")]

    if shuffle:
        random.shuffle(files)

    if limit:
        files = files[:limit]

    num_files = len(files)
    train_frac, val_frac = split

    train_size = int(train_frac * num_files)
    val_size = int(val_frac * num_files)

    train_files = files[0 : train_size]
    val_files = files[train_size : train_size + val_size]

    return train_files, val_files


def make_torch_loader(dataset, files, config, mode='val'):
    
    if dataset == 'pascal':
        dataset = PascalDataset(
                files,
                output_size=(config.segmenter._HEIGHT, config.segmenter._WIDTH),
                scale=(config.segmenter._MIN_SCALE, config.segmenter._MAX_SCALE))
        
        
    dataset.means = torch.load(ROOT + 'datasets/pascal/means.pt')
    dataset.use_transforms = True
    
    if mode == 'train':
        dataset.is_training = True
        
    loader_args = {
        "batch_size": config.segmenter._BATCH_SIZE,
        "shuffle": True,
        "drop_last": True,
        "pin_memory" : True}
    
    return torch.utils.data.DataLoader(dataset, **loader_args)

