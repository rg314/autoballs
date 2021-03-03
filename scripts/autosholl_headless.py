#@Context context
#@LogService log
"""
file:       Sholl_Extract_Profile_From_Image_Demo.py
author:     Tiago Ferreira
info:       Demonstrates how to programmatically obtain a Sholl profile from a 
            segmented image
"""
from autoballs.ijconn import get_imagej_obj


ij = get_imagej_obj(fiji_path='/home/ryan/Documents/Fiji.app', headless=False)

from scyjava import jimport
import matplotlib.pyplot as plt 

# from sc.fiji.snt.analysis.sholl import (Profile, ShollUtils)
Profile = jimport('sc.fiji.snt.analysis.sholl.Profile')
ShollUtils = jimport('sc.fiji.snt.analysis.sholl.ShollUtils')

# from sc.fiji.snt.analysis.sholl.parsers import (ImageParser2D, ImageParser3D)
ImageParser2D = jimport('sc.fiji.snt.analysis.sholl.parsers.ImageParser2D')
ImageParser3D = jimport('sc.fiji.snt.analysis.sholl.parsers.ImageParser3D')

def main(imp, headless=False):

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
        from ij import IJ
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
    parser.setRadii(10, 5, parser.maxPossibleRadius()) # (sr, er, ss)

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
        rad = list(ij.py.from_java(profile.radii()))
        cnt = list(ij.py.from_java(profile.counts()))
        plt.scatter(rad, cnt)
        plt.show()
        profile.plot().show()
    
    return profile

# For this demo we are going to use the ddaC sample image
demo_image = ShollUtils.sampleImage()
# demo_image.show()
profile = main(demo_image, headless=False)
