import matplotlib
matplotlib.use('Agg')

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

PATH = None


def config(sample, path=PATH):
    configs = dict()
    
    if not path:
        if sys.platform == "linux" or sys.platform == "linux2":
            path = '/home/ryan/Google Drive/TFM Cambridge/2021/Frogs'
        elif sys.platform == "win32":
            path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'
        elif sys.platform == 'darwin':
            path = '/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
        

    configs['path'] = path
    configs['sample'] = sample
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

def main(configs):
    best_model = torch.hub.load_state_dict_from_url('https://docs.google.com/uc?export=download&id=13CLZoNyvCt2K46UvAyHUqH7099FbnBh_')
    if configs['device'] == 'cpu':
        best_model = best_model.to('cpu')
    else:
        best_model = best_model.to('cuda')
    preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')
    preprocessing_fn = get_preprocessing(preproc_fn)


    if configs['sholl']:
        ij_obj = get_imagej_obj(headless=configs['headless'])

    files = glob.glob(f"{configs['path']}/{configs['sample']}/**/series.nd2", recursive=True)

    log = defaultdict(list)
    for idx, file in enumerate(files):
        list_of_images = imread(file)

        if list_of_images:
            for idx_img, image in enumerate(list_of_images):
                print(image.metadata['pixel_microns'])
                # get metadata and target directory
                frog_metadata = list(map(configs['frog_metadata'].get, filter(lambda x:x in file, configs['frog_metadata'])))
                gel_metadata = list(map(configs['gel_metadata'].get, filter(lambda x:x in file.lower(), configs['gel_metadata'])))[-1]
                parent_dir = os.path.basename(os.path.dirname(file))

                # filter image with fft bandpass
                filtered = fft_bandpass_filter(image, clache=False)

                # locate eyeball and return cnts
                eyeball, cnt = locate_eyeball(image)

                # if segmentation use cnn to segment image
                if configs['seg']:
                    target = get_mask(
                        filtered, 
                        model=best_model, 
                        pre_fn=get_preprocessing(preproc_fn), 
                        device=configs['device'])
                    target = ~np.array(target*255, dtype='uint8')
                else:
                    target = filtered

                # center eyeball
                centered = center_eyeball(target, cnt)

                # if sholl analysis in config run sholl
                if configs['sholl']:
                    sholl_df, sholl_mask, profile = sholl_analysis(
                        centered, 
                        ij_obj, 
                        cnt=cnt,
                        step_size=configs['step_size'], 
                        headless=configs['headless'],
                        )        
                

                    # get the median axon length 
                    median_axon_pixel = mediun_axon_length_pixels(sholl_df)
                    median_axon_um = median_axon_pixel * image.metadata['pixel_microns']
                    

                    print(f'Doing {idx+1} out of {len(files)} for stack {idx_img} out of {len(list_of_images)}. {gel_metadata} mediun axon length: {median_axon_um}')

                    # target files for mask and csv
                    mask_target = configs['results_path'] + f'/masks/{parent_dir}_s{idx_img}.tif'
                    csv_target = configs['results_path'] + f'/csvs/{parent_dir}_s{idx_img}.csv'

                    # if paths don't exist make them
                    tpath, _ = os.path.split(mask_target)
                    if not os.path.exists(tpath):
                        os.makedirs(tpath) 

                    tpath, _ = os.path.split(csv_target)
                    if not os.path.exists(tpath):
                        os.makedirs(tpath)                    

                    # write middle data
                    centered[centered<255] = 0
                    cv2.imwrite(mask_target, centered)
                    sholl_df.to_csv(csv_target)
                    
                    # create output df based on summary results
                    log['Gel type'].append(gel_metadata)
                    log['Median axon'].append(median_axon_um)
                    log['File'].append(f'{parent_dir}_s{idx_img}')
                    log['Frog meta'].append(frog_metadata)

                    res_df = pd.DataFrame(log)
                    res_df.to_csv(configs['results_path']+'/results.csv')

                # save processing steps as matplotlib
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
                    
                    target_file = configs['results_path'] + f'/imgs/{parent_dir}_s{idx_img}.png'
                    tpath, _ = os.path.split(target_file)
                    if not os.path.exists(tpath):
                        os.makedirs(tpath)
                    plt.savefig(target_file)
                    plt.close()
    
    # calcualte stats and save
    view_dataset_results(res_df.dropna())
    plt.savefig(configs['results_path']+'.jpeg') 
    plt.close()


# get configs
# configs = config('20210226 Cam Franze')
# print(configs)
# main(configs)

targets = ['20210219 Cam Franze', '20210226 Cam Franze', '20210305 Cam Franze', '20210312 Cam Franze']


for target in targets:
# run for target sample
    try:
        configs = config(target)
        print(configs)
        main(configs)
    except:
        continue