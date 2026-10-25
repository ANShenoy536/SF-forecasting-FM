"""
This program gets the predictions of all the folds for all the models into a single file
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from TrainNEval_Pt import *
from PT_DataGeneratorsFits import CustDatGen_1ch
from PtGPU_helper import *

import torch
from torch.utils.data import DataLoader
import sys

if (len(sys.argv) == 2:
    cls = argv[1]
else:
    print('Enter the command line input as: \n\t\tGenInput_5fold_Avg.py cls')
    print('cls: "Flare" or "NoFlare"')
    sys.exit(0)

device = get_default_device()
print('Device: ',device)

df_lab = pd.read_csv('../split_csv/FiveFoldCV.csv')

all_models = {'fm_name': list of model_num for fold-wise best model}
# Eg. {'Jz': [17,7,18,13,2], 'USFLUX': [10,7,4,8,10], 'LOS': [1,7,13,15,2], 'USJH': [14,8,7,7,2], 'TOTPOT': [9,11,11,13,13]}
test_dir_list = {'Jz': ['../data/Jz_preprocessed'], 
                 'USFLUX': ['../data/USFLUX_preprocessed'], 
                 'LOS': ['../data/LOS_preprocessed'], 
                 'USJH': ['../data/USJH_preprocessed'],
                 'TOTPOT': ['../data/TOTPOT_preprocessed']}
extensions_list = {'Jz': ['.jz.fits'], 'USFLUX': ['.usflux.fits'], 'LOS': ['.magnetogram.fits'], 'USJH': ['.USJH.fits'], 'TOTPOT': ['.TOTPOT.fits']}

batch_size = 32
df_final = pd.DataFrame()
for sav_nm in all_models:
   print('========================== Save name: ', sav_nm)
   #print(f'Models in {sav_nm}: {models[sav_nm]}')
   df_all = pd.DataFrame()
   arr = np.array([])
   fold = np.array([])
   models = all_models[sav_nm]
   for i in range(len(models)):
      num = models[i]
      Set = i + 1 
      df = pd.read_csv('../Outputs/' + sav_nm + '/Set' + str(Set) + 'Pred_csv/ResNet18_' + sav_nm + '_pred_' + str(num) + '.csv')
      df_all = pd.concat([df_all, df], axis = 0)
      arr = np.concatenate((arr,df_lab[df_lab['Fold'] == Set ]['Label'].values), axis = 0)
      fold = np.concatenate((fold, df_lab[df_lab['Fold'] == Set ]['Fold'].values), axis = 0)

   df_final[sav_nm] = df_all[f'P({cls})'].values

df_final['Label'] = arr.astype('int')
df_final['Fold'] = fold.astype('int')
print(df_final)
if not os.path.exists('csv_inps/'):
    os.makedirs('csv_inps/')
df_final.to_csv(f'csv_inps/1chPredEnsemble_{cls}.csv', header = True, index = False)
