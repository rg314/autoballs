import os
import sys

import glob
import matplotlib.pyplot as plt

from autoballs.utils import biometa, imread, make_figure, fft_bandpass_filter, locate_eyeball



if sys.platform == "linux" or sys.platform == "linux2":
    path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'
elif sys.platform == "win32":
    path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'



SAMPLE = '20210226 Cam Franze'
METADATA_FILE = f'{path}/{SAMPLE}{os.sep}metadata.txt'
METADATA = biometa(METADATA_FILE)


FROG_METADATA = METADATA['frog']
GEL_METADATA = METADATA['gel']

files = glob.glob(f'{path}/{SAMPLE}/**/series.nd2', recursive=True)


for file in files:
    list_of_images = imread(file)

    if list_of_images:
        for image in list_of_images:


            filtered = fft_bandpass_filter(image)
            eyeball, cnt = locate_eyeball(image)

    
            fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    
            ax[0].imshow(filtered, cmap='gray')
            ax[1].imshow(eyeball, cmap='gray')
            plt.show()







# for file in files:
#     if 'series.nd2' in file:
#         frog_metadata = list(map(FROG_METADATA.get, filter(lambda x:x in file, FROG_METADATA)))
#         gel_metadata = list(map(GEL_METADATA.get, filter(lambda x:x in file.lower(), GEL_METADATA)))
#         list_of_images = imread(file)
#         fig = make_figure(list_of_images)
#         plt.suptitle(f'{frog_metadata[0]}\n{gel_metadata[0]}',  size=50)
#         # plt.show()
#         # plt.savefig()
#         name = file.split(os.sep)[-2] + '.pdf'
#         plt.show

# # # plt.title()
# # # plt.show()



