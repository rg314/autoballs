import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utlis import get_img_from_seg, visualize
from dataloader import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing

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


# need to update path
DATA_DIR = 'data_tile/..'
DIR = DATA_DIR.split('/')[0]

ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background'] #['background', 'cell', 'balls']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
TRAIN = True
BATCH_SIZE = 16
SIZE = 256
image_types = 'images'


images = glob.glob(f'{DIR}/{image_types}/*.tif')
masks = [x.replace(f'{image_types}/img_', 'masks/mask_') for x in images]

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
    augmentation=get_validation_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
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


for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask.T
    )