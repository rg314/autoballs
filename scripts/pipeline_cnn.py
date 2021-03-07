import os
import sys

import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import segmentation_models_pytorch as smp

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

from autoballs.network.dataloader import get_preprocessing
from autoballs.network.segment import get_mask
from autoballs.ijconn import get_imagej_obj
import autoballs

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def config():
    configs = dict()
    
    if sys.platform == "linux" or sys.platform == "linux2":
        path = '/media/ryan/9684408684406AB7/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    elif sys.platform == "win32":
        path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'
    elif sys.platform == 'darwin':
        path = '/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    

    configs['path'] = path
    configs['sample'] = '20210226 Cam Franze' #'20210226 Cam Franze'
    configs['metadata_file'] = f"{configs['path']}/{configs['sample']}{os.sep}metadata.txt"
    configs['metadata'] = biometa(configs['metadata_file'])
    configs['frog_metadata'] = configs['metadata']['frog']
    configs['gel_metadata'] = configs['metadata']['gel']
    configs['sholl'] = True
    configs['create_results'] = True
    configs['results_path'] = 'results' + os.sep + configs['sample'] + '_results'
    configs['seg'] = True
    configs['headless'] = True
    configs['step_size'] = 5
    configs['device'] = 'cpu'
    configs['best_model'] = './best_model_1.pth'

    if configs['create_results']:
        if not os.path.exists(configs['results_path']):
            os.makedirs(configs['results_path'])

    return configs

def main():
    blockPrint()
    configs = config()

    best_model = torch.load(configs['best_model'])
    if configs['device'] == 'cpu':
        best_model = best_model.to('cpu')
    preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')
    preprocessing_fn = get_preprocessing(preproc_fn)


    if configs['sholl']:
        ij_obj = get_imagej_obj(headless=configs['headless'])

    files = glob.glob(f"{configs['path']}/{configs['sample']}/**/series.nd2", recursive=True)
    files = [x for x in files if '/ve' not in x]
    files = [x for x in files if 'fg1' in x]

    log = defaultdict(list)
    for idx, file in enumerate(files):
        list_of_images = imread(file)

        if list_of_images:
            for idx_img, image in enumerate(list_of_images):
                frog_metadata = list(map(configs['frog_metadata'].get, filter(lambda x:x in file, configs['frog_metadata'])))
                gel_metadata = list(map(configs['gel_metadata'].get, filter(lambda x:x in file.lower(), configs['gel_metadata'])))[-1]
                parent_dir = os.path.basename(os.path.dirname(file))

                filtered = fft_bandpass_filter(image, clache=False)

                eyeball, cnt = locate_eyeball(image)

                if configs['seg']:
                    target = get_mask(
                        filtered, 
                        model=best_model, 
                        pre_fn=get_preprocessing(preproc_fn), 
                        device=configs['device'])
                    target = ~np.array(target*255, dtype='uint8')
                else:
                    target = filtered
                centered = center_eyeball(target, cnt)

                if configs['sholl']:
                    sholl_df, sholl_mask, profile = sholl_analysis(
                        centered, 
                        ij_obj, 
                        cnt=cnt,
                        step_size=configs['step_size'], 
                        headless=configs['headless'],
                        )        
                
                    median_axon_pixel = mediun_axon_length_pixels(sholl_df)
                    median_axon_um = median_axon_pixel * image.metadata['pixel_microns']
                    
                    enablePrint()
                    print(f'Doing {idx} out of {len(files)} for stack {idx_img} out of {len(list_of_images)}. {gel_metadata} mediun axon length: {median_axon_um}')
                    blockPrint()
                    cv2.imwrite(configs['results_path'] + f'/{parent_dir}_s{idx_img}.tif', centered)
                    sholl_df.to_csv(configs['results_path'] + f'/{parent_dir}_s{idx_img}.csv')
                    
                    log['Gel type'].append(gel_metadata)
                    log['Median axon'].append(median_axon_um)

                    res_df = pd.DataFrame(log)
                    res_df.to_csv(configs['results_path']+'/res.csv')

                if configs['create_results'] and configs['sholl']:
                    images =dict(image=image, locate_eye=eyeball, filter_center=centered, sholl=sholl_mask)

                    n = len(images)
                    fig = plt.figure(figsize=(16, 5))
                    for i, (name, image) in enumerate(images.items()):
                        cmap = 'gray' if name != 'sholl' else 'jet' 
                        plt.subplot(1, n, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title(' '.join(name.split('_')).title())
                        plt.imshow(image,cmap=cmap)
                        plt.tight_layout()
                        
                    plt.savefig(configs['results_path'] + f'/{parent_dir}_s{idx_img}.png')
                    plt.close()
    

    view_dataset_results(res_df)
    plt.savefig(configs['results_path']+'/results.jpeg') 
    plt.close()



main()