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

device = get_default_device()
print('Device: ',device)

df_lab = pd.read_csv('/home/nitk233sp001/Preprocessing/70_20_10/FiveFoldCV.csv')

all_models = {'fm_name': list of model_num for fold-wise best model}
# Eg. {'Jz': [17,7,18,13,2], 'USFLUX': [10,7,4,8,10], 'LOS': [1,7,13,15,2], 'USJH': [14,8,7,7,2], 'TOTPOT': [9,11,11,13,13]}
test_dir_list = {'Jz': ['../data/Jz_preprocessed'], 
                 'USFLUX': ['../data/USFLUX_preprocessed'], 
                 'LOS': ['../data/LOS_preprocessed'], 
                 'USJH': ['../data/USJH_preprocessed'],
                 'TOTPOT': ['../data/TOTPOT_preprocessed']}
extensions_list = {'Jz': ['.jz.fits'], 'USFLUX': ['.usflux.fits'], 'LOS': ['.magnetogram.fits'], 'USJH': ['.USJH.fits'], 'TOTPOT': ['.TOTPOT.fits']}
class_weights_all = {'Set1': torch.FloatTensor([0.0323,0.9677]),
                     'Set2': torch.FloatTensor([0.0307,0.9693]),
                     'Set3': torch.FloatTensor([0.0430,0.9570]),
                     'Set4': torch.FloatTensor([0.0602,0.9398]),
                     'Set5': torch.FloatTensor([0.0478,0.9522])}

batch_size = 32
print('In the redone results, the pred_csv files donot have the probs. but the actual values. Therefore we need a similar file as below for negative data as well (at least for the average ensemble). That is why I am running this code again but for making the file with NoFlare values')

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

   df_final[sav_nm] = df_all['P(NoFlare)'].values

df_final['Label'] = arr.astype('int')
df_final['Fold'] = fold.astype('int')
print(df_final)
if not os.path.exists('csv_inps/'):
    os.makedirs('csv_inps/')
df_final.to_csv('csv_inps/1chPredEnsemble_noflare.csv', header = True, index = False)
