import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


size = 512
tile_size = (size, size)
offset = (size, size)
image_types = 'data'

origin_path = os.path.abspath(os.path.join(f'train_data/{image_types}/*', os.pardir))

images = glob.glob(origin_path+'/imgs/*.tif')
masks = [x.replace(f'imgs/img_', 'masks/img_') for x in images]

data = [(x, y) for (x, y) in list(zip(images, masks)) if os.path.exists(x) and os.path.exists(y)]
images, masks = zip(*data)


x = []
y = []
idx = 0
non_zero = 0
for img_n, mask_n in list(zip(images, masks)):

    mask = cv2.imread(mask_n)

    mask = (mask > 255//2) * 255

    img = cv2.imread(img_n)

    mask = np.asarray(mask).astype('uint8')
    mask = mask[:,:,0]

    img_shape = img.shape

    # cv2.imwrite('test.tif', mask)
    # print(mask)


    if mask.shape[:2] == img.shape[:2]:

        for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
            for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
                cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
                cropped_mask = mask[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
                #

                path = os.getcwd() + f'/train_data/data_tile_{size}/imgs'
                if not os.path.exists(path):
                    os.makedirs(path)
                imtgt = 'img_'+str(idx).zfill(5)+'.tif'
                img_target = os.path.join(path, imtgt)


                path = os.getcwd() + f'/train_data/data_tile_{size}/masks'
                if not os.path.exists(path):
                    os.makedirs(path)
                mskgt = imtgt
                mask_target = os.path.join(path, mskgt)



                # # print(cropped_img.shape, img_target)
                # # print(cropped_mask.shape, mask_target)


                cv2.imwrite(img_target, cropped_img)
                cv2.imwrite(mask_target, ~cropped_mask)


                if np.sum(cropped_mask) > 0:
                    non_zero += 1
                
                idx += 1
                
                print(f'Total {non_zero} out of {idx} which is {(non_zero*100/idx):.2f} %')
    