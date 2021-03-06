import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from autoballs.utils import get_img_from_seg, visualize
from autoballs.network.dataloader import (
    Dataset, 
    get_training_augmentation, 
    get_validation_augmentation,
    get_test_augmentation, 
    get_preprocessing,
)

import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

import os
import glob
import cv2
import random
import matplotlib.pyplot as plt
import math

# need to update path
DATA_DIR = 'train_data/..'
DIR = DATA_DIR.split('/')[0]

ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background'] #['background', 'cell', 'balls']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
TRAIN = True
VIS = False
BATCH_SIZE = 4
SIZE = 512
IMAGE_TYPE = 'data_tile_512'
IN_CHANNELS=1
MODEL_NAME = 'best_model_1.pth'


# get images and maks
images = glob.glob(f'{DIR}/{IMAGE_TYPE}/imgs/*.tif')
masks = [x.replace(f'/imgs/img_', '/masks/img_') for x in images]

# filter to make sure that they exist
data = [(x, y) for (x, y) in list(zip(images, masks)) if os.path.exists(x) and os.path.exists(y)]
images, masks = zip(*data)


images = images[:100]
masks = masks[:100]

x_test_dir = images
y_test_dir = masks


# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=IN_CHANNELS
)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
loss = smp.utils.losses.JaccardLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])



# load best saved checkpoint
best_model = torch.load('./best_model.pth')



# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_test_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    in_channels=IN_CHANNELS,
    test=True,
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)


logs = test_epoch.run(test_dataloader)

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)


images = glob.glob(f'{DIR}/data/imgs/*.tif')

preproc_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing_fn = get_preprocessing(preproc_fn)

size = 512
tile_size = (size, size)
offset = (size, size)

template = np.zeros((1024,1024,1))

for img_n in images:
    img = cv2.imread(img_n)
    img = np.asarray(img)[:,:,:3]

    img_shape = img.shape


    image = preprocessing_fn(image=img)['image'][:IN_CHANNELS,:,:]
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(pr_mask)
    plt.savefig('test.png')
    plt.close()

# for n in range(100):
    
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
    
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
#     visualize(
#         image=image_vis, 
#         ground_truth_mask=gt_mask, 
#         predicted_mask=pr_mask
#     )
#     plt.savefig('test.png')
