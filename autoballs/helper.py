"""
Thank you to jzrolling for some helper functions for the FFT
https://github.com/jzrolling/OMEGA
OMEGA is a python based microscopic image analysis toolkit optimized for mycobacterial cells.
03/17/2020
"""

import numpy as np
from scipy import fftpack
from skimage import exposure
import warnings



def fft(img, subtract_mean=True):
    """
    fast Fourier transform module
    :param img: input image
    :param subtract_mean:
    :return: FFT transformed image
    """
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return (fftpack.fftshift(fftpack.fft2(img)))


def fft_reconstruction(fft_img, filters):
    """
    reconstruct image after FFT band pass filtering.
    :param fft_img: input FFT transformed image
    :param filters: low/high frequency bandpass filters
    :return: bandpass filtered, restored phase contrast image
    """

    warnings.filterwarnings("ignore")
    if len(filters) > 0:
        for filter in filters:
            try:
                fft_img *= filter
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    return (fftpack.ifft2(fftpack.ifftshift(fft_img)).real)



def bandpass_filter(pixel_microns, img_width=2048, img_height=2048, high_pass_width=0.2, low_pass_width=20):

    """

    :param pixel_microns: pixel unit length
    :param img_width: width of image by pixel
    :param img_height: height of image by pixel
    :param high_pass_width: 1/f where f is the lower bound of high frequency signal
    :param low_pass_width: 1/f where f is the upper bound of low frequency signal
    :return: high/low bandpass filters

    """
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    if high_pass_width == 0:
        high_pass_filter = np.ones((img_width, img_height)).astype(np.int)
    else:
        high_pass_filter = np.e ** (-(centered_mesh * high_pass_width) ** 2)
    if low_pass_width == 0:
        low_pass_filter = np.ones((2048, 2048)).astype(np.int)
    else:
        low_pass_filter = 1 - np.e ** (-(centered_mesh * low_pass_width) ** 2)
    return (high_pass_filter, low_pass_filter)



def is_integer(x):
    try:
        isinstance(x, (int))
        return True
    except:
        return False



def adjust_image(img, dtype=16, adjust_gamma=True, gamma=1):
    """
    adjust image data depth and gamma value
    :param img: input image
    :param dtype: bit depth, 8, 12 or 16
    :param adjust_gamma: whether or not correct gamma
    :param gamma: gamma value
    :return: adjusted image
    """
    if is_integer(dtype) & (dtype > 2):
        n_range = (0, 2 ** dtype - 1)
    else:
        print("Illegal input found where an integer no less than 2 was expected.")
    outimg = exposure.rescale_intensity(img, out_range=n_range)
    if adjust_gamma:
        outimg = exposure.adjust_gamma(outimg, gamma=gamma)
    return outimg

