###############################################
# Downloading different segments of SHARP data
#----------------------------------------------
# Saving the segments from each sample in dirs
# named HARPNUM__T_REC.
###############################################

import os
import drms
from sunpy.net import Fido
from astropy import units as u
from sunpy.net import attrs as a
import pandas as pd
from tqdm import tqdm

save_dir = "./data/raw_data"
email_id = '<YOUR_EMAIL>' # This email needs to be registered with JSOC (ref http://jsoc.stanford.edu/ajax/register_email.html)

client = drms.Client()

df= pd.read_csv("./split_csv/FiveFoldCV.csv")

start = 0
end = len(df)

for i in tqdm(range(start,end), total = end-start):
    harp = df['HARP no.'].values[i]
    T = pd.Timestamp(df['T_REC'].values[i])
        
    
    folder_name = str(harp) + '__' +  str(T).replace(' ','_').replace(':','').replace('-','')
    save_loc = os.path.join(save_dir,folder_name)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    result = Fido.search(a.Time(T, T),
                     a.jsoc.Series('hmi.sharp_cea_720s'),
                     a.jsoc.PrimeKey("HARPNUM", harp),
                     a.jsoc.Segment('magnetogram') & a.jsoc.Segment('Bp') & a.jsoc.Segment('Br') & a.jsoc.Segment('Bt') & a.jsoc.Segment('bitmap'),
                     a.jsoc.Notify(email_id))
    dwld_fl= Fido.fetch(result,path=save_loc,overwrite=False)

print("\nAll Done!")
