import os
import sys

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from autoballs.utils import (biometa, 
                            imread, 
                            make_figure, 
                            fft_bandpass_filter, 
                            locate_eyeball, 
                            center_eyeball, 
                            sholl_analysis,
                            visualize,
                            )

import autoballs.helper as helper
from autoballs.ijconn import get_imagej_obj


if sys.platform == "linux" or sys.platform == "linux2":
    path = '/media/ryan/9684408684406AB7/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
elif sys.platform == "win32":
    path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'



SAMPLE = '20210226 Cam Franze'
METADATA_FILE = f'{path}/{SAMPLE}{os.sep}metadata.txt'
METADATA = biometa(METADATA_FILE)


FROG_METADATA = METADATA['frog']
GEL_METADATA = METADATA['gel']

SHOLL = False


if SHOLL:
    ij_obj = get_imagej_obj()

files = glob.glob(f'{path}/{SAMPLE}/**/series.nd2', recursive=True)


for file in files:
    list_of_images = imread(file)

    if list_of_images:
        for image in list_of_images:
            frog_metadata = list(map(FROG_METADATA.get, filter(lambda x:x in file, FROG_METADATA)))
            gel_metadata = list(map(GEL_METADATA.get, filter(lambda x:x in file.lower(), GEL_METADATA)))

            filtered = fft_bandpass_filter(image)
            eyeball, cnt = locate_eyeball(image)
            centered = center_eyeball(filtered, cnt)


            if SHOLL:
                sholl_df, sholl_mask = sholl_analysis(centered, ij_obj)        
            
        
            # images =dict(image=image, locate_eye=eyeball, filter_center=centered, sholl=sholl_mask)
            # fig = visualize(show=False, **images)
            # plt.savefig('example_proc.png')
            # plt.close()


