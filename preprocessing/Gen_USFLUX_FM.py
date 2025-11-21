#################################
# # Generating the FM of USFLUX
#################################
# This file generates the FM for the unsigned magnetic flux from the pre-processed  vector magnetograms - B_phi, B_r, B_theta


import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from astropy.io import fits
import sys

if len(sys.argv) == 3:
    ffcsv = argv[1]
    src_dir = argv[2]
else: 
    print('Enter the command line input: \n\t\tGen_USFLUX_FM.py ffcsv src_dir')
    sys.exit(0)

#Load the dataframe containing the dataset - FiveFoldCV.csv
df = pd.read_csv(ffcsv)

# directories to the vector magnetograms
dirs = [os.path.join(src_dir, "Bp_preprocessed"), 
        os.path.join(src_dir, "Br_preprocessed"),
        os.path.join(src_dir, "Bt_preprocessed")]
extensions = ['.Bp.fits', '.Br.fits','.Bt.fits']

# destination directory for USFLUX pre-processed files
dest_pth = os.path.join(src_dir, "\USFLUX_preprocessed")

if not os.path.exists(dest_pth):
    os.makedirs(dest_pth)

for fl in tqdm(df['FileName'], total = len(df)):
    ar_p = fits.getdata(os.path.join(dirs[0], fl) + extensions[0]) #load_img(os.path.join(dirs[0], fl) + extensions[0])
    ar_r = fits.getdata(os.path.join(dirs[1], fl) + extensions[1])
    ar_t = fits.getdata(os.path.join(dirs[2], fl) + extensions[2])
    
    usflux = ((ar_p ** 2) + (ar_r ** 2) + (ar_t ** 2))/3
    usflux = np.sqrt(usflux)
    
    #cv2.imwrite(os.path.join(dest, fl) + '.usflux.png', usflux)
    hdu = fits.PrimaryHDU(usflux)
    hdu.writeto(os.path.join(dest, fl) + '.usflux.fits')



