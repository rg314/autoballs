import os
import cv2
import tempfile
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from nd2reader import ND2Reader
from scipy.signal import convolve2d

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

    # apply clache to enhance contrast
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)[:,:,0]

    # threshold
    threshold = cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,6)

    cv2.imwrite('test.tif', threshold)
    plt.imshow(threshold)
    plt.show()

    return threshold

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
    centered_image = cv2.warpAffine(~image, T, (width, height))

    return ~centered_image



def sholl_analysis(img, ij_obj):
    filter_img = img
    radius = np.max(img.shape) // 2

    name = os.urandom(24).hex()
    tmp_file = os.path.join(tempfile.gettempdir(), name) + '.tif'

    cv2.imwrite(tmp_file,filter_img)


    script = f"""importClass(Packages.ij.IJ)
            importClass(Packages.ij.gui.PointRoi)
            imp = IJ.openImage("{tmp_file}");
            //IJ.setAutoThreshold(imp, "Default dark");
            //IJ.run(imp, "Convert to Mask", "");
            //IJ.setTool("multipoint");
            IJ.run(imp, "Convert to Mask", "");
            imp.setRoi(new PointRoi({radius},{radius},"small yellow hybrid"));
            imp.show();
            IJ.run(imp, "Sholl Analysis...", "starting=10 ending={radius} radius_step=0 #_samples=5 integration=Mean enclosing=1 #_primary=0 infer fit linear polynomial=[Best fitting degree] most normalizer=Area create overlay save directory={tempfile.gettempdir()}");
            IJ.run("Close All", "");
            """
    
    args = {}
    ij_obj.py.run_script('js', script, args)

    tmp_csv = os.path.join(tempfile.gettempdir(), name) + '_Sholl-Profiles.csv'
    sholl_df = pd.read_csv(tmp_csv)
    
    tmp_mask = os.path.join(tempfile.gettempdir(), name) + '_ShollMask.tif'
    sholl_mask = cv2.imread(tmp_mask, -1)
    constant = (255-0)/(sholl_mask.max()-sholl_mask.min()) 
    img_stretch = sholl_mask * constant 
    sholl_mask = cv2.applyColorMap(img_stretch.astype('uint8'), cv2.COLORMAP_HSV)


    centered = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay_mask = cv2.addWeighted(centered,1,sholl_mask,1,0)

    return sholl_df, overlay_mask

def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)



def mediun_axon_length_pixels(df, cnt):
    #min circle (green)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    rad, inters = df[['Radius', 'Inters.']].T.values
     
    inters = np.trim_zeros(inters)
    trim_zeros = len(rad) - len(inters)
    rad = rad[trim_zeros:]
    rad = rad - rad[0] 

    output = []
    i =0 
    ones = np.array([])
    while i < len(rad):
        ones = np.append(ones, np.ones(int(inters[i]))*rad[i])
        i+=1
    med = np.median(ones)

    return med

    