import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from nd2reader import ND2Reader

import autoballs.helper as helper


def imread(file):
    with ND2Reader(file) as images:
        img = images
        list_of_images = []
        for i in range(len(img)):
            list_of_images.append(img[i])
            

    return list_of_images


def biometa(metadata):
    with open(metadata, 'r') as f:
        lines = f.readlines()
    return eval(''.join([x.strip('\n') for x in lines]))


def make_figure(list_of_images):
    fig=plt.figure(figsize=(16, 16))
    rows = 2
    columns = len(list_of_images) // rows
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        if list_of_images:
            image = list_of_images.pop(0)
            plt.imshow(image, cmap='gray')
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    return fig

def fft_bandpass_filter(image, pixel_microns = 1024 / 1331.2, bandpass_high = 12, bandpass_low = 120, gamma=1, normalize=True):

    img_fft = helper.fft(image)
    fft_filters = helper.bandpass_filter(pixel_microns=pixel_microns,
                                            img_width=image.shape[1], img_height=image.shape[0],
                                            high_pass_width=bandpass_high,
                                            low_pass_width=bandpass_low)

    fft_reconstructed = helper.fft_reconstruction(img_fft, fft_filters)

    if normalize:
        fft_reconstructed = helper.adjust_image(fft_reconstructed, adjust_gamma=True, gamma=gamma)

    # 0-255 and uint8 dtype 
    data = fft_reconstructed
    data *= (255.0/data.max())
    data = data.astype('uint8')

    # apply clache 
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)[:,:,0]

    threshold = cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,6)

    return threshold



def locate_eyeball(image):
    image = ~(image>image.min()*2) * 1
    contours,_ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # rank areas
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)

    #bounding box (red)
    cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour

    #fit ellipse (blue)
    ellipse = cv2.fitEllipse(cnt)
    image = cv2.cvtColor(np.array(image*255, dtype='uint8'), cv2.COLOR_GRAY2RGB)
    cv2.ellipse(image,ellipse,(255,0,0),2)
    return image, cnt


def get_img_from_seg(path_to_file):
	path, file = os.path.split(path_to_file)
	img_name = f'{os.sep}'.join(path.split(os.sep)[:-1]).split('_')[0] + os.sep+ file.replace('man_seg', 't')
	if os.path.exists(img_name):
		return img_name
	else:
		msg = 'Raw image not found'
		raise ValueError(msg)

# helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
