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
VIS = False
BATCH_SIZE = 4
SIZE = 256
image_types = 'images'


images = glob.glob(f'{DIR}/{image_types}/*.tif')
masks = [x.replace(f'{image_types}/img_', 'masks/mask_') for x in images]

print(images)
print(masks)

X = images
y = masks

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)


if VIS:
    
    train_dataset = Dataset(
        x_train,
        y_train,
        augmentation=get_training_augmentation(size=SIZE), 
        # preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    for i in range(100):
        image, mask = train_dataset[i]
        visualize(image=image, mask=mask)




# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    x_train,
	y_train,
    augmentation=get_training_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid,
	y_valid,
    augmentation=get_validation_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# for i in range(10):
#     image, mask = train_dataset[10]
#     visualize(image=image, mask=mask)


if TRAIN:

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.JaccardLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])


    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )



    # train model for 40 epochs

    max_score = 0

    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
