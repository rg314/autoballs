import os
import cv2
import tempfile
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from nd2reader import ND2Reader
from scipy.signal import convolve2d


import seaborn as sns
import statsmodels.stats.multicomp as multi

import autoballs.helper as helper
from scyjava import jimport


def imread(file):
    with ND2Reader(file) as images:
        img = images
        list_of_images = []
        for i in range(len(img)):
            list_of_images.append(img[i])
            

    return list_of_images


def biometa(metadata):
    try:
        with open(metadata, 'r') as f:
            lines = f.readlines()
        return eval(''.join([x.strip('\n') for x in lines]))
    except:
        return None

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

def kuwahara_filter(image, kernel_size=5):
    """桑原フィルターを適用した画像を返す
    https://github.com/Kazuhito00/Kuwahara-Filter
    Args:
        image: OpenCV Image
        kernel_size: Kernel size is an odd number of 5 or more

    Returns:
        Image after applying the filter.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    height, width, channel = image.shape[0], image.shape[1], image.shape[2]

    r = int((kernel_size - 1) / 2)
    r = r if r >= 2 else 2

    image = np.pad(image, ((r, r), (r, r), (0, 0)), "edge")

    average, variance = cv2.integral2(image)
    average = (average[:-r - 1, :-r - 1] + average[r + 1:, r + 1:] -
               average[r + 1:, :-r - 1] - average[:-r - 1, r + 1:]) / (r +
                                                                       1)**2
    variance = ((variance[:-r - 1, :-r - 1] + variance[r + 1:, r + 1:] -
                 variance[r + 1:, :-r - 1] - variance[:-r - 1, r + 1:]) /
                (r + 1)**2 - average**2).sum(axis=2)

    def filter(i, j):
        return np.array([
            average[i, j], average[i + r, j], average[i, j + r], average[i + r,
                                                                         j + r]
        ])[(np.array([
            variance[i, j], variance[i + r, j], variance[i, j + r],
            variance[i + r, j + r]
        ]).argmin(axis=0).flatten(), j.flatten(),
            i.flatten())].reshape(width, height, channel).transpose(1, 0, 2)

    filtered_image = filter(*np.meshgrid(np.arange(height), np.arange(width)))

    filtered_image = filtered_image.astype(image.dtype)
    filtered_image = filtered_image.copy()

    return filtered_image


def fft_bandpass_filter(image, pixel_microns = 1024 / 1331.2, bandpass_high = 12, bandpass_low = 120, gamma=1, normalize=True, clache=True):

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

    if clache:
        # apply clache to enhance contrast
        img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)[:,:,0]

        # threshold
        data = cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,6)

    return data


def segment(image, eyeball):
    # denoise while preserving edges
    kuwahara = kuwahara_filter(image, 3)[:,:,0]

    # dilate
    dilate1 = cv2.dilate(~kuwahara,np.ones((5,5),np.uint8),iterations = 5)

    # merge
    merge_eyeball = dilate1 + eyeball[:,:,0]

    # dilate
    dilate2 = cv2.dilate(merge_eyeball,np.ones((5,5),np.uint8),iterations = 5)

    # threshold
    threshold = (dilate2 > 200) * 255

    # find largest blob
    contours,_ = cv2.findContours(threshold.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)

    #bounding box (red)
    cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour

    # mask image
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1, cv2.LINE_AA)
    target = cv2.bitwise_and(kuwahara, kuwahara, mask=mask)
    target = target + ~mask
    
    return target

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
def visualize(show=False, **images):
    """Plot images in one row."""
    n = len(images)
    fig = plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if len(image.shape)==2:
            plt.imshow(image,cmap='gray')
        else:
            plt.imshow(image)

    plt.tight_layout()
    
    if show:
        plt.show()

    return fig 


def center_eyeball(image, cnt):
    ellipse = cv2.fitEllipse(cnt)
    centerE = ellipse[0]

    height, width = image.shape
    wi=(width/2)
    he=(height/2)

    cX = centerE[0]
    cY = centerE[1]

    offsetX = (wi-cX)
    offsetY = (he-cY)
    T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
    centered_image = cv2.warpAffine(~image.astype('uint8')*255, T, (width, height))

    return ~centered_image



def sholl_analysis(img, ij_obj, cnt=None, starting_radius=0, step_size=5, headless=True):
    """
    Thank you Tiago Ferreira for the input
    https://forum.image.sc/t/automated-sholl-analysis-headless/49601
    https://github.com/morphonets/SNT/blob/master/src/main/resources/script_templates/Neuroanatomy/Analysis/Sholl_Extract_Profile_From_Image_Demo.py
    """
    if cnt.any() != None:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        starting_radius = int(radius)


    ij = ij_obj
    imp = ij.py.to_java(img)
    
    ImagePlusClass    = jimport('ij.ImagePlus')
    imp = ij.dataset().create(imp)
    imp = ij.convert().convert(imp, ImagePlusClass)
    


    # from sc.fiji.snt.analysis.sholl import (Profile, ShollUtils)
    Profile = jimport('sc.fiji.snt.analysis.sholl.Profile')
    ShollUtils = jimport('sc.fiji.snt.analysis.sholl.ShollUtils')
    ImageParser2D = jimport('sc.fiji.snt.analysis.sholl.parsers.ImageParser2D')
    ImageParser3D = jimport('sc.fiji.snt.analysis.sholl.parsers.ImageParser3D')
    

    # We may want to set specific options depending on whether we are parsing a
    # 2D or a 3D image. If the image has multiple channels/time points, we set
    # the C,T position to be analyzed by activating them. The channel and frame
    # will be stored in the profile properties map and can be retrieved later):
    if imp.getNSlices() == 1:
        parser = ImageParser2D(imp)
        parser.setRadiiSpan(0, ImageParser2D.MEAN) # mean of 4 measurements at every radius
        parser.setPosition(1, 1, 1) # channel, frame, Z-slice
    else: 
        parser = ImageParser3D(imp)
        parser.setSkipSingleVoxels(True) # ignore isolated voxels
        parser.setPosition(1, 1) # channel, frame
  
    # Segmentation: we can set the threshold manually using one of 2 ways:
    # 1. manually: parser.setThreshold(lower_t, upper_t)
    # 2. from the image itself: e.g., IJ.setAutoThreshold(imp, "Huang")
    # If the image is already binarized, we can skip setting threshold levels:
    if not (imp.isThreshold() or imp.getProcessor().isBinary()):
        IJ = jimport('ij.IJ')
        IJ.setAutoThreshold(imp, "Otsu dark")

    # Center: the x,y,z coordinates of center of analysis. In a real-case usage
    # these would be retrieved from ROIs or a centroid of a segmentation routine.
    # If no ROI exists coordinates can be set in spatially calibrated units
    # (floats) or pixel coordinates (integers):
    if imp.getRoi() is None:
        xc = int(round(imp.getWidth()/2))
        yc = int(round(imp.getHeight()/2))
        zc = int(round(imp.getNSlices()/2))
        parser.setCenterPx(xc, yc, zc)  # center of image
    else:
        parser.setCenterFromROI()

    # Sampling distances: start radius (sr), end radius (er), and step size (ss).
    # A step size of zero would mean 'continuos sampling'. Note that end radius
    # could also be set programmatically, e.g., from a ROI
    parser.setRadii(starting_radius, step_size, parser.maxPossibleRadius()) # (sr, ss, er)

    # We could now set further options as we would do in the dialog prompt:
    parser.setHemiShells('none')
    # (...)

    # Parse the image. This may take a while depending on image size. 3D images
    # will be parsed using the number of threads specified in ImageJ's settings:
    parser.parse()
    if not parser.successful():
        log.error(imp.getTitle() + " could not be parsed!!!")
        return

    # We can e.g., access the 'Sholl mask', a synthetic image in which foreground
    # pixels have been assigned the no. of intersections:
    if not headless:
        parser.getMask().show()

    # Now we can access the Sholl profile:
    profile = parser.getProfile()
    if profile.isEmpty():
        log.error("All intersection counts were zero! Invalid threshold range!?")
        return

    # We can now access all the measured data stored in 'profile': Let's display
    # the sampling shells and the detected sites of intersections (NB: If the
    # image already has an overlay, it will be cleared):
    profile.getROIs(imp)
    
    # For now, lets's perform a minor cleanup of the data and plot it without
    # doing any polynomial regression. Have a look at Sholl_Extensive_Stats_Demo
    # script for details on how to analyze profiles with detailed granularity
    profile.trimZeroCounts()

    if not headless:
        profile.plot().show()

    sholl_df = pd.DataFrame(
        {
            'Radius': list(ij.py.from_java(profile.radii())), 
            'Inters.': list(ij.py.from_java(profile.counts()))
        }
    )

    sholl_df = filter_discontinuity(sholl_df)
    
    mask = np.array(ij.py.from_java(parser.getMask()))


    return sholl_df, mask, profile

def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)



def mediun_axon_length_pixels(sholl_df):
    rad, inters = sholl_df[['Radius', 'Inters.']].T.values
    rad = rad - rad[0]

    output = []
    i =0 
    ones = np.array([])
    while i < len(rad):
        ones = np.append(ones, np.ones(int(inters[i]))*rad[i])
        i+=1
    med = np.median(ones)

    return med


def view_dataset_results(data):    
    sns.set_style("white")
    sns.set_style("ticks")
    ax = sns.boxplot(y='Median axon', x='Gel type', data=data, palette="Blues")
    ax = sns.swarmplot(y='Median axon', x='Gel type', data=data, color=".25", size=10)
    ax.set_ylabel('Axon length [um]')
    ax.set_xlabel('Gel type [kPa]')

    test = multi.MultiComparison(data['Median axon'], data['Gel type'])
    res = test.tukeyhsd()
    res_table1 = res.summary()
    print(res_table1)


    test = multi.pairwise_tukeyhsd(data['Median axon'], data['Gel type'], alpha=0.05)
    res_table2 = test.summary()
    print(res_table2)

    # with open(txt_path, 'w') as f:
        # f.write()

    return ax

def getMaxLength(arr): 
    n = len(arr)
    count = 0 
    result = 0 
  
    for i in range(0, n): 
        if (arr[i] != 1): 
            count = 0
        else: 
            count+= 1 
            result = max(result, count)  
          
    return result  

def find_subsequence(seq, subseq):
    if not subseq.any():
        return 0
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq,
                                       subseq, mode='valid') == target)[0]
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask][0]


def filter_discontinuity(df):
        rad, inters = df[['Radius', 'Inters.']].T.values

        # find longest 1s
        ans = getMaxLength(inters)
        target = np.ones(ans)
        # find index where longest ones start
        idx = find_subsequence(inters, target)

        # zero discontinuity
        inters[idx:] = 0

        # update df
        df['Radius'] = rad
        df['Inters.'] = inters

        return df