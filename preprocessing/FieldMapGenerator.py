# This is a modified version of ``calculate_sharpkeys.py `` from:
# 
# **SHARPs**  
# https://github.com/mbobra/SHARPs.git \
# 
# Used under the MIT License.  
## ---- Original License Notice (MIT) ----
# Copyright (c) 2019 Monica Bobra
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


###############################################
# Generating the field maps of physical features
###############################################
# This file generates the field maps or spatial distributions of the vertical current (Jz), unsigned current helicity (USFLUX) and the photospheric magnetic free energy density (TOTPOT). It also generates their error files. 
# 
# This requires the following SHARP segments:
#            example filename                   --> description
#            hmi.sharp_cea_*.Br.fits            --> radial component of the magnetic field vector \
#            hmi.sharp_cea_*.Bt.fits            --> theta-component of the magnetic field vector \
#            hmi.sharp_cea_*.Bp.fits            --> phi-component of the magnetic field vector \
#            hmi.sharp_cea_*.Br_err.fits        --> error in radial component of the magnetic field vector \
#            hmi.sharp_cea_*.Bt_err.fits        --> error in theta-component of the magnetic field vector \
#            hmi.sharp_cea_*.Bp_err.fits        --> error in phi-component of the magnetic field vector \
#            hmi.sharp_cea_*.conf_disambig.fits --> bits indicate confidence levels in disambiguation result \
#            hmi.sharp_cea_*.bitmap.fits        --> bits indicate result of automatic detection algorithm \
#            hmi.sharp_cea_*.magnetogram.fits   --> line-of-sight component of the magnetic field \
# 
# Note: it okay to skip generating the error files.



from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from tqdm import tqdm
import os
import sunpy.map
from skimage.measure import block_reduce
import scipy.ndimage
import imageio
from skimage import exposure
from datetime import datetime, timedelta
import json
import csv
import sys

import warnings
warnings.simplefilter('ignore')



if len(sys.argv) == 3:
	base_dir = sys.argv[1]
	dest_dir = sys.argv[2]
else:
	print('Enter the command line input as: \n\t\tFieldMapGenertor.py base_dir dest_dir')
	sys.exit(0)


radsindeg = np.pi/180.
munaught  = 0.0000012566370614

noaas = os.listdir(base_dir)
noaas.sort()
tot_files=len(noaas)
print('Total Active region folder: ',tot_files)
skip_list=[]

loop_idx=0

def update_skipFile(file,miss_par): # accounts for skipped file
    with open("skip_files.txt", "a") as f:
        f.write( file+' '+miss_par+"\n")

# os.remove('skip_files.txt') #For fresh start, clears the previous file

jz_dest = os.path.join(dest_dir, "Jz")
jz_dest_err = os.path.join(dest_dir, "Jz_err")

jh_dest = os.path.join(dest_dir, "USJH")
jh_dest_err = os.path.join(dest_dir, "USJH_err")

totpot_dest = os.path.join(dest_dir, "TOTPOT")
totpot_dest_err = os.path.join(dest_dir, "TOTPOT_err")

json_path_dest = os.path.join(dest_dir, "JSON_Files")

for noaa in noaas:
	loop_idx+=1
	if noaa.isdigit():

		times = os.listdir(os.path.join(base_dir,noaa))
		times.sort()
		tot_time_stamps=len(times)
		print()
		print(f"Calculating parameters for NOAA {noaa} [ {loop_idx} / {tot_files} ]")
			

		for tstamp in tqdm(times):

			ts_dir = os.path.join(os.path.join(base_dir,noaa),tstamp)
			
			if os.path.isdir(ts_dir):


				files = os.listdir(ts_dir)
				files.sort()

				inc=azi=fld=cds=bmp=inc_err=azi_err=fld_err=mgm = 0

				for f in files:
					f = os.path.join(ts_dir,f)
					if f.endswith('Bt.fits'):
						inc = f
					elif f.endswith('Bp.fits'):
						azi = f
					elif f.endswith('Br.fits'):
						fld = f
					elif f.endswith('conf_disambig.fits'):
						cds = f
					elif f.endswith('bitmap.fits'):
						bmp = f
					elif f.endswith('Bt_err.fits'):
						inc_err = f
					elif f.endswith('Bp_err.fits'):
						azi_err = f
					elif f.endswith('Br_err.fits'):
						fld_err = f
					elif f.endswith('magnetogram.fits'):
						mgm = f

					elif f.endswith('.fits.1'):
						os.remove(f)
				#print(inc,mgm,fld_err,azi_err,inc_err,bmp,cds,fld,azi)
				if inc ==0:
					print('Missing inclination file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'incination')
					continue
				if mgm ==0:
					print('Missing magnetogram file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'magnetogram')
					continue

				if fld_err ==0:
					print('Missing field error file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'field error')
					continue

				if azi_err ==0:
					print('Missing azimuth error file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'azimuth error')
					continue

				if inc_err ==0:
					print('Missing inclination error file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'inclination error')
					continue

				if bmp ==0:
					print('Missing bitmap file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'bitmap')
					continue

				if cds ==0:
					print('Missing conf disambig file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'conf disambig')
					continue

				if fld ==0:
					print('Missing field file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'field')
					continue

				if azi ==0:
					print('Missing azimuth file')
					print('Skipping it')
					skip_list.append(ts_dir)
					update_skipFile(ts_dir,'azimuth')
					continue


				bz_map = sunpy.map.Map(fld)
				by_map = sunpy.map.Map(inc)
				bx_map = sunpy.map.Map(azi)
				bz_err_map = sunpy.map.Map(fld_err)
				by_err_map = sunpy.map.Map(inc_err)
				bx_err_map = sunpy.map.Map(azi_err)
				bmp_map = sunpy.map.Map(bmp)
				cds_map = sunpy.map.Map(cds)
				los_map = sunpy.map.Map(mgm)

				
				'''
				bz		 = bz_map.data
				by		 = by_map.data
				bx		 = bx_map.data
				bz_err			= bz_err_map.data
				by_err			= by_err_map.data
				bx_err			= bx_err_map.data
				conf_disambig	= cds_map.data
				bittmap		   = bmp_map.data
				los			   = los_map.data
				'''

				# Creating common header file for all fits file
				with fits.open(fld) as hdul_bz:
					bz = hdul_bz[0].data if hdul_bz[0].data is not None else hdul_bz[1].data
					bz_header = hdul_bz[1].header if hdul_bz[1].header is not None else hdul_bz[0].header

					#--For proper Header information--
					req_header_keys=['DATE-OBS','T_REC ','T_OBS ','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','CUNIT1','CUNIT2','IMCRPIX1','IMCRPIX2','IMCRVAL1','IMCRVAL2','CROTA2','WCSNAME','DSUN_OBS','DSUN_REF','RSUN_REF','CRLN_OBS','CRLT_OBS','CAR_ROT','RSUN_OBS','LAT_MIN','LON_MIN','LAT_MAX','LAT_MAX','LON_MAX','N_PATCH','N_PATCH1','N_PATCHM','NOAA_AR','NOAA_NUM','NOAA_ARS',]

					emt_ar=np.empty_like(bz[1].data)
					empty_fits=fits.PrimaryHDU(emt_ar)
					for  key_ in req_header_keys:
						empty_fits.header[key_]=bz_header[key_]
					file_headers=empty_fits.header
				#---------//\\------#

				with fits.open(inc) as hdul_by:
					by = hdul_by[0].data if hdul_by[0].data is not None else hdul_by[1].data

				with fits.open(azi) as hdul_bx:
					bx = hdul_bx[0].data if hdul_bx[0].data is not None else hdul_bx[1].data

				with fits.open(fld_err) as hdul_bze:
					bz_err = hdul_bze[0].data if hdul_bze[0].data is not None else hdul_bze[1].data
					bz_err_header = hdul_bze[1].header if hdul_bze[1].header is not None else hdul_bze[0].header

				with fits.open(inc_err) as hdul_bye:
					by_err = hdul_bye[0].data if hdul_bye[0].data is not None else hdul_bye[1].data

				with fits.open(azi_err) as hdul_bxe:
					bx_err = hdul_bxe[0].data if hdul_bxe[0].data is not None else hdul_bxe[1].data

				with fits.open(cds) as hdul_cds:
					conf_disambig = hdul_cds[0].data if hdul_cds[0].data is not None else hdul_cds[1].data

				with fits.open(mgm) as hdul_mgm:
					los = hdul_mgm[0].data if hdul_mgm[0].data is not None else hdul_mgm[1].data

				with fits.open(bmp) as hdul_bmp:
					bitmap = hdul_bmp[0].data if hdul_bmp[0].data is not None else hdul_bmp[1].data

				nx	 = bz.shape[1]
				ny	 = bz.shape[0]

				header = bz_map.meta
				rsun_ref = header['rsun_ref'] # reference radius of the sun in m
				cdelt1	 = header['cdelt1'] # map scale in X direction in degrees
				dsun_obs = header['dsun_obs'] # distance from SDO to sun center in m
				rsun_obs = header['rsun_obs'] # angular radius of the sun in arcsec (corresponding to the satellite SDO)


				cdelt1_arcsec = (math.atan((rsun_ref*cdelt1*radsindeg)/(dsun_obs)))*(1/radsindeg)*(3600.)
				los_err = np.ndarray(shape=(ny,nx), dtype=float)
				los_err.fill(6.4)
				by_flipped = -1.0*(np.array(by))

				def length_of_vector(vx, vy):
					length = np.sqrt(vx**2 + vy**2)
					return length


				vx = bx
				vy = by

				vector_length = length_of_vector(vx, vy)

				def angle_between_vectors(bx, by):
					theta_radians = np.arctan2(by, bx)
					theta_degrees = np.degrees(theta_radians)
					return theta_radians, theta_degrees

				# Calculating Current Density Values 
				bx = bx
				by = by

				derx	  = np.zeros([ny,nx])
				dery	  = np.zeros([ny,nx])
				err_term1 = np.zeros([ny,nx])
				err_term2 = np.zeros([ny,nx])
				jz		= np.zeros([ny,nx])
				jz_err	= np.zeros([ny,nx])

				for i in range(1,nx-1):
					for j in range(0,ny):
						derx[j,i]	  = (by[j,i+1] - by[j,i-1])*0.5
						err_term1[j,i] = ((by_err[j,i+1]*by_err[j,i+1]) + (by_err[j,i-1]*by_err[j,i-1]))
				
				for i in range(0,nx):
					for j in range(1,ny-1):
						dery[j,i]	  = (bx[j+1,i] - bx[j-1,i])*0.5
						err_term2[j,i] = ((bx_err[j+1,i]*bx_err[j+1,i]) + (bx_err[j-1,i]*bx_err[j-1,i]))
				
				i=0
				for j in range(ny):
					derx[j,i] = ( (-3*by[j,i]) + (4*by[j,i+1]) - (by[j,i+2]) )*0.5
					
				i=nx-1
				for j in range(ny):
					derx[j,i] = ( (3*by[j,i]) + (-4*by[j,i-1]) - (-by[j,i-2]) )*0.5
				
				j=0
				for i in range(nx):
					dery[j,i] = ( (-3*bx[j,i]) + (4*bx[j+1,i]) - (bx[(j+2),i]) )*0.5
				
				j=ny-1
				for i in range(nx):
					dery[j,i] = ( (3*bx[j,i]) + (-4*bx[j-1,i]) - (-bx[j-2,i]) )*0.5
				
				
				total_jz = 0
				for j in range(1,ny-1):
					for i in range (1,nx-1):
						jz[j,i]	 = (derx[j,i] - dery[j,i])
						jz_err[j,i] = 0.5*np.sqrt(err_term1[j,i] + err_term2[j,i])

				# Creating FITS for Jz 
				f_pattern = str(fld.replace("Br.fits",""))
				
# 				jz_dest = r"D:\PGStuff\SolarFlare\SolarData\New folder\Jz"
# 				jz_dest_err = r"D:\PGStuff\SolarFlare\SolarData\New folder\Jz_err"
				f_pattern_jz = str(f_pattern.replace(ts_dir, jz_dest))
				f_pattern_jz_err = str(f_pattern_jz.replace('Jz', 'Jz_err'))
				jz_fits = f_pattern_jz+'jz'+'.fits'
				jz_err_fits = f_pattern_jz_err+'jz_err'+'.fits'
                
# 				print(jz_fits)
# 				print(jz_err_fits)

				

				jz_hdu = fits.PrimaryHDU(jz, header=file_headers)
				jz_hdu.writeto(jz_fits, overwrite=True)

				jz_err_hdu = fits.PrimaryHDU(jz_err, header=bz_err_header)
				jz_err_hdu.writeto(jz_err_fits, overwrite=True)

				#print(f"Jz_err values stored as FITS file in : {ts_dir} \t\t\t TS: {datetime.now()}")

				#Calculating Mean and Total Helicity

				count_mask	  = 0.0
				sum			 = 0.0
				sum2			= 0.0
				err			 = 0.0
				US_JH_img=np.zeros((ny,nx))
				JH_err_img=np.zeros((ny,nx))
			

				for j in range(ny):
					for i in range (nx):
						if ( conf_disambig[j,i] < 70 or bitmap[j,i] < 30 ):
							continue
						if np.isnan(jz[j,i]):
							continue
						if np.isnan(bz[j,i]):
							continue
						if (jz[j,i] == 0):
							continue		
						if (bz[j,i] == 0):
							continue		
						if np.isnan(jz_err[j,i]):
							continue
						if np.isnan(bz_err[j,i]):
							continue
						sum		+= (jz[j,i]*bz[j,i])*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref)	#contributes to MEANJZH and ABSNJZH
						sum2    += abs(jz[j,i]*bz[j,i])*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref) # contributes to TOTUSJH
						err		+= (jz_err[j,i]*jz_err[j,i]*bz[j,i]*bz[j,i]) + (bz_err[j,i]*bz_err[j,i]*jz[j,i]*jz[j,i])
						count_mask += 1
						total_jz += 	abs(jz[j,i])*(cdelt1_arcsec/1)*(rsun_ref/rsun_obs)*(0.00010)*(1/munaught)
						US_JH_img[j,i]=abs(jz[j,i]*bz[j,i])*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref) 
						JH_err_img[j,i]=(jz_err[j,i]*jz_err[j,i]*bz[j,i]*bz[j,i]) + (bz_err[j,i]*bz_err[j,i]*jz[j,i]*jz[j,i])


				mean_ih		  = sum/count_mask															   # Units are G^2 / m ; keyword is MEANJZH
				total_us_ih	  = sum2																		 # Units are G^2 / m ; keyword is TOTUSJH
				total_abs_ih	 = abs(sum)																	 # Units are G^2 / m ; keyword is ABSNJZH
				mean_ih_err	  = (np.sqrt(err)/count_mask)*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref) # error in the quantity MEANJZH
				total_us_ih_err  = (np.sqrt(err))*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref)			# error in the quantity TOTUSJH
				total_abs_ih_err = (np.sqrt(err))*(1/cdelt1_arcsec)*(rsun_obs/rsun_ref)			# error in the quantity ABSNJZH
				
				
# 				jh_dest = r"D:\PGStuff\SolarFlare\SolarData\New folder\USJH"
# 				jh_dest_err = r"D:\PGStuff\SolarFlare\SolarData\New folder\USJH_err"
				f_pattern_jh = str(f_pattern.replace(ts_dir, jh_dest))
				f_pattern_jh_err = str(f_pattern_jh.replace('USJH', 'USJH_err'))
                
				us_fits = f_pattern_jh+'USJH'+'.fits'
				bh_err_fits = f_pattern_jh_err+'USJH_err'+'.fits'

				JH_hdu = fits.PrimaryHDU(US_JH_img, header=file_headers)
				JH_hdu.writeto(us_fits, overwrite=True)

				JH_er_hdu = fits.PrimaryHDU(JH_err_img, header=file_headers)
				JH_er_hdu.writeto(bh_err_fits, overwrite=True)

				#print('Saved the Unsigned Currenrt Helcity files')



				# Calculating Free Energy


				count_mask	  = 0.0
				sum			 = 0.0
				sum1			= 0.0
				err			 = 0.0

				def greenpot(bz, nx, ny):
					
					nnx = nx
					nny = ny
					
					dz = 0.001

					pfpot	  = np.zeros([nny,nnx])
					rdist	  = np.zeros([nny,nnx])
					bztmp	  = np.zeros([nny,nnx])
					bxp		= np.zeros([nny,nnx])
					byp		= np.zeros([nny,nnx])

					for iny in range(nny):
						for inx in range(nnx):
							if np.isnan(bz[iny,inx]):
								bztmp[iny,inx] = 0.0
							else:
								bztmp[iny,inx] = bz[iny,inx]

					rdd  = 0.0
					rdd1 = 0.0
					rdd2 = 0.0
					for iny in range(nny):
						for inx in range(nnx):
							rdd1  = float(inx)
							rdd2  = float(iny)
							rdd   = rdd1 * rdd1 + rdd2 * rdd2 + dz * dz
							rdist[iny,inx] = 1.0/(np.sqrt(rdd))

					iwindow = 0
					if (nnx > nny):
						iwindow = nnx
					else:
						iwindow = nny

					rwindow = float(iwindow)
					rwindow = rwindow * rwindow + 0.01 # must be square	
					rwindow = 1.0e2					# limit the window size to be 10.	
					rwindow = np.sqrt(rwindow)
					iwindow = int(rwindow)

					for iny in range(nny):
						for inx in range(nnx):
							if np.isnan(bz[iny,inx]):
								pfpot[iny,inx] = 0.0
							else:
								sum = 0.0
								j2s = iny - iwindow
								j2e = iny + iwindow
								if (j2s < 0):
									j2s = 0
								if (j2e > nny):
									j2e = nny
								i2s = inx - iwindow
								i2e = inx + iwindow
								if (i2s < 0):
									i2s = 0
								if (i2e > nnx):
									i2e = nnx
								for j2 in range(j2s,j2e):
									for i2 in range(i2s,i2e):
										val1 = bztmp[j2,i2]
										di = abs(i2 - inx)
										dj = abs(j2 - iny)
										sum = sum + val1 * rdist[dj,di] * dz
								pfpot[iny,inx] = sum
					
					for iny in range(1,nny-1):
						for inx in range(1,nnx-1):
							bxp[iny,inx] = -(pfpot[iny,inx+1] - pfpot[iny,inx-1])*0.5
							byp[iny,inx] = -(pfpot[iny+1,inx] - pfpot[iny-1,inx])*0.5

					return [bxp, byp]

				potential = greenpot(bz, nx, ny)
				bpx, bpy  = potential[0], potential[1]
				tot_pot_img=np.zeros((ny,nx))
				tot_pot_img_er=np.zeros((ny,nx))

				for j in range(ny):
					for i in range (nx):
						if ( conf_disambig[j,i] < 70 or bitmap[j,i] < 30 ):
							continue
						if np.isnan(bx[j,i]):
							continue
						if np.isnan(by[j,i]):
							continue
						sum  += ( ((bx[j,i] - bpx[j,i])*(bx[j,i] - bpx[j,i])) + ((by[j,i] - bpy[j,i])*(by[j,i] - bpy[j,i])) )*(cdelt1_arcsec*cdelt1_arcsec*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0)
						sum1 += (  ((bx[j,i] - bpx[j,i])*(bx[j,i] - bpx[j,i])) + ((by[j,i] - bpy[j,i])*(by[j,i] - bpy[j,i])) )
						err  += 4.0*(bx[j,i] - bpx[j,i])*(bx[j,i] - bpx[j,i])*(bx_err[j,i]*bx_err[j,i]) + 4.0*(by[j,i] - bpy[j,i])*(by[j,i] - bpy[j,i])*(by_err[j,i]*by_err[j,i])
						count_mask += 1
						tot_pot_img[j,i]=( ((bx[j,i] - bpx[j,i])*(bx[j,i] - bpx[j,i])) + ((by[j,i] - bpy[j,i])*(by[j,i] - bpy[j,i])) )*(cdelt1_arcsec*cdelt1_arcsec*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0)
						tot_pot_img_er[j,i]=4.0*(bx[j,i] - bpx[j,i])*(bx[j,i] - bpx[j,i])*(bx_err[j,i]*bx_err[j,i]) + 4.0*(by[j,i] - bpy[j,i])*(by[j,i] - bpy[j,i])*(by_err[j,i]*by_err[j,i])

				# Units of meanpotptr are ergs per centimeter
				meanpot	  = (sum1) / (count_mask*8.*np.pi)
				meanpot_err  = (np.sqrt(err)) / (count_mask*8.*np.pi)
				
				# Units of sum are ergs/cm^3, units of factor are cm^2/pix^2; therefore, units of totpotptr are ergs per centimeter
				totpot	   = (sum)/(8.*np.pi)
				totpot_err   = (np.sqrt(err))*abs(cdelt1_arcsec*cdelt1_arcsec*(rsun_ref/rsun_obs)*(rsun_ref/rsun_obs)*100.0*100.0*(1/(8.*np.pi)))

                
# 				totpot_dest = r"D:\PGStuff\SolarFlare\SolarData\New folder\TOTPOT"
# 				totpot_dest_err = r"D:\PGStuff\SolarFlare\SolarData\New folder\TOTPOT_err"
				f_pattern_tp = str(f_pattern.replace(ts_dir, totpot_dest))
				f_pattern_tp_err = str(f_pattern_tp.replace('TOTPOT', 'TOTPOT_err'))
				totpot_fits = f_pattern_tp+'TOTPOT'+'.fits'
				totpot_err_fits = f_pattern_tp_err+'TOTPOT_err'+'.fits'

				TOTPOT_hdu = fits.PrimaryHDU(tot_pot_img, header=file_headers)
				TOTPOT_hdu.writeto(totpot_fits, overwrite=True)

				TOTPOT_er_hdu = fits.PrimaryHDU(tot_pot_img_er, header=file_headers)
				TOTPOT_er_hdu.writeto(totpot_err_fits, overwrite=True)


				json_data = {
				"mean_current_helicity" : mean_ih,
				"total_unsigned_current_helicity" : total_us_ih,
				"absolute_current_helicity" : total_abs_ih,
				"mean_current_helicity_err" : mean_ih_err,
				"total_unsigned_current_helicity_err" : total_us_ih_err,
				"total_unsigned_current_helicity_err" : total_abs_ih_err,
				"mean_photospheric_magnetic_free_energy" : meanpot,
				"mean_photospheric_magnetic_free_energy_error" : meanpot_err,
				"total_photospheric_magnetic_free_energy" : totpot,
				"total_photospheric_magnetic_free_energy_error": totpot_err
				}

				
# 				json_path = os.path.join(ts_dir,tstamp)
# 				json_path_dest = r"D:\PGStuff\SolarFlare\SolarData\New folder\JSON_Files"
				json_path = str(f_pattern.replace(ts_dir,json_path_dest))+"calculated_params.json"
# 				print(json_path)
				with open(json_path, "w") as outfile: 
					json.dump(json_data, outfile, sort_keys=False, indent=4)
					#print(f"Done Data for NOAA {noaa} [ {loop_idx} / {tot_files} ]",end='', flush=True)
					#print()
					#print(f" All the Current and Magnetic Properties for NOAA {noaa} in {json_path} ")
				
				
					


			else:
				print(f" Some or all Files of NOAA {noaa} are Missing for timestamp {tstamp}.")





