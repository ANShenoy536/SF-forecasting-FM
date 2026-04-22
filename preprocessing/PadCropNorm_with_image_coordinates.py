"""
This code carries out Pad-crop and normalization while also calculating and updating the coordinates of the pre-processed file in its FITS header.
NOTE:  this file requires the creation of the Bitmap_centroids.csv file using Get_Bitmap_Centroids.py.
NOTATION: axis = 0 => the change will be to the rows or relating to the y-axis
            axis = 1 => the change will be to the columns or relating to x-axis

"""
from astropy.io import fits

import sunpy
import sunpy.map

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm


def nan_check(img):
    img[np.isnan(img)] = 0
    return img


def padding(img, axis, fin_dim):
    if axis == 0: #pad along y or rows
        pad_img = np.zeros((fin_dim, img.shape[1]))
    elif axis == 1: #pad along x or columns
        pad_img = np.zeros((img.shape[0], fin_dim))
    x, y = pad_img.shape    
    
    pad_img[(x - img.shape[0])//2: (x + img.shape[0])//2, (y - img.shape[1])//2:(y + img.shape[1])//2] = img 
    #:img.shape[0], :img.shape[1]] = img
    
    return pad_img


def cropping(img, axis, cent_cord, fin_dim = 896):
    val = int(fin_dim/2)
    if cent_cord - val < 0:
        #print('case1')
        cent_cord = val
        #print(cent_cord)
    elif axis*(cent_cord + val > img.shape[1]): #if the cropped image would exceed the x-length of the image
        #print('case2')
        cent_cord = img.shape[1] - val
        #print(cent_cord)
    elif (1-axis)*(cent_cord + val >img.shape[0]): #if the cropped image would exceed the y-length of the image
        #print('case3')
        cent_cord = img.shape[0] - val
        #print(cent_cord)
        
    #print('cent_cord: ', cent_cord)
    if axis == 1:
        crop_img = img[:,cent_cord - val:cent_cord + val]
    elif axis == 0:
        crop_img = img[cent_cord - val:cent_cord + val,:]
    return crop_img, cent_cord


def normalize(img, max_val, min_val):
    n_img = np.zeros(img.shape)
    p_img = np.zeros(img.shape)
    
    p_img[img > 0] = (img[img > 0] - min_val)/(max_val - min_val)
    p_img[p_img < 0] = 0
    p_img[p_img > 1] = 1
    
    n_img[img < 0] = (img[img < 0] + min_val)/(max_val - min_val)
    n_img[n_img < -1] = -1
    n_img[n_img > 0] = 0
    
    return p_img + n_img



def log_normalize(img, max_val, min_val): # For TOTPOT
    n_img = np.zeros(img.shape)
    p_img = np.zeros(img.shape)
    
    p_img[img > 0] = (np.log(img[img > 0]) - np.log(min_val))/(np.log(max_val) - np.log(min_val))
    p_img[p_img < 0] = 0
    p_img[p_img > 1] = 1
    
    n_img[img < 0] = (np.log(img[img < 0]) + np.log(min_val))/(np.log(max_val) - np.log(min_val))
    n_img[n_img < -1] = -1
    n_img[n_img > 0] = 0
    
    return p_img + n_img

def get_updated_crval(header, cent_cord, axis):
    # Note that longitude= CRVAL1 is the x axis and latitude=CRVAL2 is the y axis
    # Also note that this function calculates the cval i.t.o the 896x896 image with cdelt=0.3
    if axis == 0: # y axis; latitude
        crval = header['CRVAL2'] + (cent_cord + 0.5 - header['CRPIX2'])*header['CDELT2'] # +0.5 because +1 to match sharp cord. -0.5 to get to patch center
    if axis == 1: # x axis; longitude
        crval = header['CRVAL1'] + (cent_cord + 0.5 - header['CRPIX1'])*header['CDELT1'] # +0.5 because +1 to match sharp cord. -0.5 to get to patch center
    return crval


def save_img_fits(img, dest_dir, fl_name, header, case, cent_cord_x = None, cent_cord_y = None):
    """
    cent_cord_x: (int) the x coordinate (column or axis 1) of the pixel around which the original image was cropped. Needed while commputing
                        crvalx
    cent_cord_x: (int) the y coordinate (row or axis 0) of the pixel around which the original image was cropped. Needed while commputing
                        crvaly
    """
    hdu_prim = fits.PrimaryHDU()
    hdu_img = fits.ImageHDU(img)

    if case == 1: # pad along x and y
        # Lat and Lan of the cent. of the image is same. Only change the CRPIXi along both dimensions
        # update lon or x-axis as per pad
        header['NAXIS1'] = hdu_img.header['NAXIS1']
        #update lat or y-axis as per pd
        header['NAXIS2'] = hdu_img.header['NAXIS2']
        
        # update center pixel to center of img. This is the center of the final 224x224 image.
        header['CRPIX1'] = 112.5 # 224/2 + 0.5 because it's the center of the image in the sharp coords
        header['CRPIX2'] = 112.5 # 224/2 + 0.5 because it's the center of the image in the sharp coords
        
        
    elif case == 2: # pad x and crop y
        # update the x axis (axis = 1, header i=1) as per padding
        header['NAXIS1'] = hdu_img.header['NAXIS1']
        
        # update y axis (axis = 0, header i = 2) as per cropping
        header['NAXIS2'] = hdu_img.header['NAXIS2']
        updated_crval2 = round(get_updated_crval(header, cent_cord_y, axis = 0),7)
        header['CRVAL2'] = updated_crval2
        
        # update center pixel to center of img
        header['CRPIX1'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        header['CRPIX2'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        
        
    elif case == 3: # pad y and crop x
        #update the x axis as per crop
        header['NAXIS1'] = hdu_img.header['NAXIS1']
        updated_crval1 = round(get_updated_crval(header, cent_cord_x, axis = 1),7)
        header['CRVAL1'] = updated_crval1
        
        #update the y axis as per pad
        header['NAXIS2'] = hdu_img.header['NAXIS2']
        
        # update center pixel to center of img
        header['CRPIX1'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        header['CRPIX2'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        
        
    elif case == 4: # crop along x and y
        #update the x axis as per crop
        header['NAXIS1'] = hdu_img.header['NAXIS1']
        updated_crval1 = round(get_updated_crval(header, cent_cord_x, axis = 1),7)
        header['CRVAL1'] = updated_crval1
        
        # update y axis (axis = 0, header i = 2) as per cropping
        header['NAXIS2'] = hdu_img.header['NAXIS2']
        updated_crval2 = round(get_updated_crval(header, cent_cord_y, axis = 0),7)
        header['CRVAL2'] = updated_crval2       
        
        # update center pixel to center of img
        header['CRPIX1'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        header['CRPIX2'] = 112.5 # 224/2 + 0.5 because it's the center of the center pixel
        
        
    else:
        print(f"case = {case}.Something went wrong!! None of the cases happened")
    
    
    hdu_img.header = hdu_img.header + header
    hdu_img.header['CDELT1'] = header['CDELT1'] * (896/224)
    hdu_img.header['CDELT2'] = header['CDELT2'] * (896/224)

    hdul = fits.HDUList([hdu_prim, hdu_img])
    hdul.writeto(os.path.join(dest_dir, fl_name), overwrite = True)


######################################
# Loading the paths to the directories and the Bitmaps_centroid.csv file
######################################

# FM-wise file extensions
all_exts = {'Bp': '.Bp.fits', 'Br': '.Br.fits', 'Bt': '.Bt.fits', 'Jz': '.jz.fits', 'USJH': '.USJH.fits',
            'LOS': '.magnetogram.fits', 'USFLUX': '.usflux.fits', 'TOTPOT': '.TOTPOT.fits'}
            
# FM-wise T_H (maxvals) and T_L values (minvals)
minvals = {'Bp': 100, 'Br': 100, 'Bt': 100, 'LOS': 100, 'Jz': 0, 'USJH': 0, 'TOTPOT': 1e17 }
maxvals = {'Bp': 2000, 'Br': 2000, 'Bt': 2000, 'LOS': 2000, 'Jz': 150, 'USJH': 0.5, 'TOTPOT': 1e20 }

csv_fl = "/home/spml_test/ash/helpers/Centroids.csv"
df = pd.read_csv(csv_fl, sep = ';')

cx = df['Cx']
cy = df['Cy']

fm_nm = 'TOTPOT'
dir_path = os.path.join("./data", fm_nm) #directory containing the file to crop
extension = all_exts[fm_nm] #extension of the file

dest_dir = os.path.join("./data", fm_nm + '_preprocessed') #directory where the cropped images will be saved

max_val = maxvals[fm_nm]
min_val = minvals[fm_nm]

size = (224, 224)
count = 0

for i in tqdm(range(len(df))):
    case=0
    fl_path = os.path.join(dir_path, df['FileName'][i] + extension)
    img = fits.getdata(fl_path)
    header = fits.open(fl_path)[1].header

    img = nan_check(img)
    
    y, x = img.shape
    
    if (x<=896) and (y<=896):
        pad_img = padding(img, axis = 0, fin_dim = 896)
        pad_img = padding(pad_img, axis = 1, fin_dim = 896)
        
        assert pad_img.shape == (896, 896), "In case 1: Shape of final image is not (896,896)"
        fin_img = pad_img # fin_image is the final image, that is to be saved
        cent_cord_x = None
        cent_cord_y = None
        case=1
        
    elif (x <= 896) and (y > 896): # row is greater and column is smaller that 896
        pad_img = padding(img, axis = 1, fin_dim = 896)
        crop_img, cent_cord_y = cropping(pad_img, axis = 0, cent_cord = cy[i])
        
        assert crop_img.shape == (896, 896), "In case 2: Shape of final image is not (896,896)"
        fin_img = crop_img
        cent_cord_x = None
        case=2
        
    elif (x > 896) and (y <= 896): #row is smaller and column is greater than 896
        pad_img = padding(img, axis = 0, fin_dim = 896)
        crop_img, cent_cord_x = cropping(pad_img, axis = 1, cent_cord = cx[i])
        
        assert crop_img.shape == (896, 896), f"In case 3: Shape of final image is not (896,896)"
        fin_img = crop_img
        cent_cord_y = None
        case=3
        
    elif (x > 896) and (y > 896):
        crop_img, cent_cord_x = cropping(img, axis = 1, cent_cord = cx[i])
        crop_img, cent_cord_y = cropping(crop_img, axis = 0, cent_cord = cy[i])
        
        assert crop_img.shape == (896, 896), f"In case 4: Shape of final image is not (896,896);b{fl_path}"
        fin_img = crop_img
        case=4 
        
    else:
        print('Case 5: ', img.shape)
        count += 1
    
    if fm_nm == 'TOTPOT':
        norm_img = log_normalize(fin_img, max_val, min_val)
    else:
        norm_img = normalize(fin_img, max_val, min_val)
    
    resized_img = cv2.resize(norm_img, size, interpolation = cv2.INTER_LINEAR)
    save_img_fits(resized_img, dest_dir, df['FileName'][i]+extension, header, case, cent_cord_x, cent_cord_y)
    
print('count = ', count)
print('Finished!')




