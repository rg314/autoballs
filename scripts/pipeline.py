import os
import sys

import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from autoballs.utils import (biometa, 
                            imread, 
                            make_figure, 
                            fft_bandpass_filter, 
                            locate_eyeball, 
                            center_eyeball, 
                            sholl_analysis,
                            visualize,
                            segment,
                            mediun_axon_length_pixels,
                            )
from autoballs.ijconn import get_imagej_obj
import autoballs


def config():
    configs = dict()
    
    if sys.platform == "linux" or sys.platform == "linux2":
        path = '/media/ryan/9684408684406AB7/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    elif sys.platform == "win32":
        path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'

    configs['path'] = path

    configs['sample'] = '20210226 Cam Franze'
    configs['metadata_file'] = f"{configs['path']}/{configs['sample']}{os.sep}metadata.txt"
    configs['metadata'] = biometa(configs['metadata_file'])
    configs['frog_metadata'] = configs['metadata']['frog']
    configs['gel_metadata'] = configs['metadata']['gel']
    configs['sholl'] = True
    configs['create_results'] = True
    configs['results_path'] = configs['sample'] + '_results'
    configs['seg'] = False

    if configs['create_results']:
        if not os.path.exists(configs['results_path']):
            os.mkdir(configs['results_path'])

    return configs

def main():
    configs = config()

    if configs['sholl']:
        ij_obj = get_imagej_obj()

    files = glob.glob(f"{configs['path']}/{configs['sample']}/**/series.nd2", recursive=True)

    log = defaultdict(list)
    for file in files:
        list_of_images = imread(file)

        if list_of_images:
            for image in list_of_images:
                frog_metadata = list(map(configs['frog_metadata'].get, filter(lambda x:x in file, configs['frog_metadata'])))
                gel_metadata = list(map(configs['gel_metadata'].get, filter(lambda x:x in file.lower(), configs['gel_metadata'])))


                filtered = fft_bandpass_filter(image)
                eyeball, cnt = locate_eyeball(image)

                if configs['seg']:
                    target = segment(filtered, eyeball)
                else:
                    target = filtered
                centered = center_eyeball(target, cnt)


                if configs['sholl']:
                    sholl_df, sholl_mask = sholl_analysis(centered, ij_obj)        
                
                median_axon_pixel = mediun_axon_length_pixels(sholl_df, cnt)
                median_axon_um = median_axon_pixel * image.metadata['pixel_microns']
                
                print(gel_metadata[0], 'mediun axon length: ', median_axon_um)
                
                log['Gel type'].append(gel_metadata[0])
                log['Median axon'].append(median_axon_um)

                res_df = pd.DataFrame(log)
                res_df.to_csv(configs['results_path']+'/res.csv')

                if configs['create_results'] and configs['sholl']:
                    pass
                    # images =dict(image=image, locate_eye=eyeball, filter_center=centered, sholl=sholl_mask)
                    # fig = visualize(show=False, **images)
                    # plt.savefig(configs['results_path'] + '/example_proc.png')
                    # plt.close()



main()