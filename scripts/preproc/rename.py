import os
import glob
import cv2
import matplotlib.pyplot as plt

images = glob.glob('data/images/*.tif')
masks = [x.replace('images/img_', 'masks/mask_') for x in images]

x = []
y = []
for mask, img in list(zip(images, masks)):
    m = cv2.imread(mask)
    i = cv2.imread(img)

    if m.shape == i.shape:
        x.append(img)
        y.append(mask)


    # fig, ax = plt.subplots()

    # ax[0].imshow(img)
    # ax[1].imshow(mask)
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