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
                            view_dataset_results,
                            )
from autoballs.ijconn import get_imagej_obj
import autoballs


def config():
    configs = dict()
    
    if sys.platform == "linux" or sys.platform == "linux2":
        raise OSError('not run')
        # path = '/media/ryan/9684408684406AB7/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    elif sys.platform == "win32":
        path = 'C:\\Users\\ryan\\Documents\\GitHub\\Easy-to-use_TFM_package\\test_data'
    elif sys.platform == 'darwin':
        path = '/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    

    configs['path'] = path

    configs['sample'] = '20210226 Cam Franze'
    configs['metadata_file'] = f"{configs['path']}/{configs['sample']}{os.sep}metadata.txt"
    configs['metadata'] = biometa(configs['metadata_file'])
    configs['frog_metadata'] = configs['metadata']['frog']
    configs['gel_metadata'] = configs['metadata']['gel']
    configs['sholl'] = False
    configs['create_results'] = False
    configs['results_path'] = 'results' + os.sep + configs['sample'] + '_results'
    configs['seg'] = False
    configs['headless'] = True
    configs['step_size'] = 50

    if configs['create_results']:
        if not os.path.exists(configs['results_path']):
            os.makedirs(configs['results_path'])

    return configs

def main():
    configs = config()

    if configs['sholl']:
        ij_obj = get_imagej_obj(headless=configs['headless'])

    files = glob.glob(f"{configs['path']}/{configs['sample']}/**/series.nd2", recursive=True)

    log = defaultdict(list)
    i = 163
    for idx, file in enumerate(files):
        list_of_images = imread(file)

        if list_of_images:
            for idx_img, image in enumerate(list_of_images):
                frog_metadata = list(map(configs['frog_metadata'].get, filter(lambda x:x in file, configs['frog_metadata'])))
                gel_metadata = list(map(configs['gel_metadata'].get, filter(lambda x:x in file.lower(), configs['gel_metadata'])))


                filtered = fft_bandpass_filter(image, clache=False)


                cv2.imwrite(f'train_data/img_{str(i).zfill(4)}.tif', filtered)
                i += 1
                print(i)

                # eyeball, cnt = locate_eyeball(image)

                # if configs['seg']:
                #     target = segment(filtered, eyeball)
                # else:
                #     target = filtered
                # centered = center_eyeball(target, cnt)


                # if configs['sholl']:
                #     sholl_df, sholl_mask, profile = sholl_analysis(
                #         centered, 
                #         ij_obj, 
                #         step_size=configs['step_size'], 
                #         headless=configs['headless'],
                #         )        
                
                #     median_axon_pixel = mediun_axon_length_pixels(sholl_df)
                #     median_axon_um = median_axon_pixel * image.metadata['pixel_microns']
                    
                #     print(gel_metadata[0], 'mediun axon length: ', median_axon_um)
                #     # cv2.imwrite(f'{idx}-{idx_img}-len{int(median_axon_um)}.png',sholl_mask)
                    
                #     log['Gel type'].append(gel_metadata[0])
                #     log['Median axon'].append(median_axon_um)

                #     res_df = pd.DataFrame(log)
                #     res_df.to_csv(configs['results_path']+'/res.csv')

                # if configs['create_results'] and configs['sholl']:
                #     pass
                #     # images =dict(image=image, locate_eye=eyeball, filter_center=centered, sholl=sholl_mask)
                #     # fig = visualize(show=False, **images)
                #     # plt.savefig(configs['results_path'] + '/example_proc.png')
                #     # plt.close()
    

    # view_dataset_results(res_df)
    # plt.savefig('results.jpeg') 



main()