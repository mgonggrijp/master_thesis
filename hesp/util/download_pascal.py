import torchvision
import os

print('Downloading pascal voc2012 data')
if not os.path.exists("datasets/pascal/data/VOCdevkit/"):
    torchvision.datasets.VOCSegmentation(
        root="datasets/pascal/data/",
        year='2012',
        image_set='train',
        download=True,
        transforms=None)
print('done..')