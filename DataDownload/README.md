# Data Download
The SHARP segments corresponding to the dataset by Hollanda et al. (2021) has been compiled in the `split_csv/FiveFoldCV.csv`, with the HARPNUMs (or HARP numbers) of sampled obtained by mapping them with NOAA AR numbers. 
To Download the data run the following script:
```python
python3 DataDownload/RawDataDownload.py
```
The above script will generate folders in the format <HARPNUM>__<T_REC> in the `data/raw_data`, which would containg the corresponding Bp, Br, Bt, magnetogram, and bitmap segments. We need to seperate these into segment_wise folders. For this run the following
```shell
cd data
mkdir Bp Br Bt LOS Bitmap
for i in raw_data/*; do
mv "${i}/*.bitmap.fits" Bitmap/
mv "${i}/*.magnetogram.fits" LOS/
mv "${i}/*.Bp.fits" Bp/
mv "${i}/*.Br.fits" Br/
mv "${i}/*.Bt.fits" Bt/
```
