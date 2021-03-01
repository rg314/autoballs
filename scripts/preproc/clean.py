import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np 

images = glob.glob('images/*.tif')
masks = [x.replace('images/img_', 'masks/mask_') for x in images]



x = []
y = []
for img, mask in list(zip(images, masks)):
    m = cv2.imread(mask)
    i = cv2.imread(img)

    m = np.asarray(~m).astype('uint8')
    m = m[:,:,0]

    m[m==255] = 1
    m[m==96] = 2
    m[m > 2] = 0

    cv2.imwrite(mask, m)

    # plt.imshow(m)
    # plt.show()
    
    # fig, ax = plt.subplots(1, 2)

    # ax[0].imshow(i)
    # ax[1].imshow(m)
    # plt.show()

# print(images)
# for idx, im in enumerate(images):
#     path, basename = os.path.split(im)
#     print(basename)
#     img = cv2.imread(im)
#     print(img.shape)
    # img_name = im
    # mask_name = img_name.replace('images', 'masks')
    # if os.path.exists(mask_name) and os.path.exists(img_name):
    #     imtgt = 'img_'+str(idx).zfill(4)+'.tif'
    #     mskgt = imtgt.replace('img_', 'mask_')

    #     path, basename = os.path.split(img_name)
    #     target = os.path.join(path, imtgt)

    #     os.rename(img_name, target)

    #     path, basename = os.path.split(mask_name)
    #     target = os.path.join(path, mskgt)

    #     os.rename(mask_name, target)