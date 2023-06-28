""" Download the COCO images and annotations from https://github.com/nightrome/cocostuff#downloads"""

import requests
import zipfile
import os
# import PIL 
# import numpy as np
# import matplotlib.pyplot as plt

print('Attempting to download COCO images and annotations from https://github.com/nightrome/cocostuff#downloads ..')

# download the validation image set
if not os.path.exists('datasets/coco/data/val2017'):
    print('Downloading coco validation images; size 1GB..')

    url = 'http://images.cocodataset.org/zips/val2017.zip'
    r = requests.get(url, allow_redirects=True)
    _ = open('val2017.zip', 'wb').write(r.content)

    with zipfile.ZipFile('val2017.zip', 'r') as zip_ref:
        zip_ref.extractall("datasets/coco/data")

    os.remove('val2017.zip')
    print('Validation images downloaded.')

else:
    print('Validation folder already present, skipping download.')


# download the training image set
if not os.path.exists('datasets/coco/data/train2017'):
    print('Downloading training images; size 18GB..')
    url = 'http://images.cocodataset.org/zips/train2017.zip'
    r = requests.get(url, allow_redirects=True)
    _ = open('train2017.zip', 'wb').write(r.content)

    with zipfile.ZipFile('train2017.zip', 'r') as zip_ref:
        zip_ref.extractall("datasets/coco/data")

    os.remove('train2017.zip')
    print('Training images donwloaded.')

else:
    print('Training image folder already found, skipping download.')


""" NOTE link not working; have to download manually from https://github.com/nightrome/cocostuff#downloads
--> Downloads --> stuffthingmaps_trainval2017.zip """
# download the stuff + thing annotations; coco stuff and things
if not os.path.exists('datasets/coco/data/stuffthingmaps_trainval2017'):
    # print('Downloading coco stuff + thing annotations; size 659MB..')
    # url = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip'
    # r = requests.get(url, allow_redirects=True)
    # _file = open('stuffthingmaps_trainval2017.zip', 'wb').write(r.content)

    print('Unpacking coco labels zip..')
    with zipfile.ZipFile('stuffthingmaps_trainval2017.zip', 'r') as zip_ref:
        zip_ref.extractall("datasets/coco/data/annotations")

    # print('Done downloading and extracting annotations.')

else:
    print('..')

