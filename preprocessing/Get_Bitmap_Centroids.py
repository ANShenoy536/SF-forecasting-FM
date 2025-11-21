#######################
#Get Bitmap Centroids
#######################
# This notebook reads from a directory of SHARP bitmaps, finds their centroids and stores them in a csv file

import numpy as np
import os
import cv2
import copy

from astropy.io import fits
import sys


# #### Functions to check the files for NaN values, and replacing them with 0

# replacing nan with 0
def replace_nan(img):
    img_file = copy.copy(img)
    ind = np.where(np.isnan(img_file) == True)
    for i in range(len(ind[0])):
        img_file[ind[0][i], ind[1][i]] = 0
    return img_file 

# checking the image for NaNs
def nan_check(img):
    if np.isnan(img).any():
        img_new = replace_nan(img)
        return img_new
    else:
        return img




if len(sys.argv) == 3:
    src_path = sys.argv[1]
    fl_path = sys.argv[2]
else:
    print("Enter the command line input as: \n\t\tGet_Bitmap_Centroids.py src_path fl_path")
    sys.exit(0)


f = open(fl_path, 'a')
f.write('FileName;Cx;Cy\n')


# Finding the centroids

for fl in os.listdir(src_path):
    img = fits.getdata(os.path.join(src_path, fl))
    
    img = nan_check(img)
    
    threshold, bin_img = cv2.threshold((img).astype('uint8'), 1, 1,0) 
    M = cv2.moments(bin_img)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    f.write(fl.replace('.bitmap.fits','') + ';' + str(cX) + ';' + str(cY) + '\n')

f.close()



