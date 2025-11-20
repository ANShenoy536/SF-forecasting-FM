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

"""
#models1 = [24,34,9,3,14]  #Vmgm
#models2 = [5,6,16,7,10] #JFP
#models3 = [8,8,4,3,15] # FP
models_jz = [4,7,4,10,4] #Jz
models5 = [9,7,6,7,2] #USFLUX
models6 = [17,8,2,14,4] #LOS
models7 = [12,8,11,10,12] #USJH
models8 = [8,7,3,9,14] #TOTPOT
#models = [19,33,2,9,4] #JFH
"""
all_models = {'Jz': [17,7,18,13,2], 'USFLUX': [10,7,4,8,10], 'LOS': [1,7,13,15,2], 'USJH': [14,8,7,7,2], 'TOTPOT': [9,11,11,13,13]}
#model_dirs = {'Jz': 'Jz', 'USFLUX': 'USFLUX', 'LOS': 'LOS', 'USJH': 'Helicity', 'TOTPOT': 'TOTPOT'} 
test_dir_list = {'Jz': ['/home/nitk233sp001/RawFM_fits/Jz2'], 
                 'USFLUX': ['/home/nitk233sp001/RawFM_fits/USFLUX2'], 
                 'LOS': ['/home/nitk233sp001/RawFM_fits/LOS2'], 
                 'USJH': ['/home/nitk233sp001/RawFM_fits/USJH2'],
                 'TOTPOT': ['/home/nitk233sp001/RawFM_fits/TOTPOT2']}
extensions_list = {'Jz': ['.jz.fits'], 'USFLUX': ['.usflux.fits'], 'LOS': ['.magnetogram.fits'], 'USJH': ['.USJH.fits'], 'TOTPOT': ['.TOTPOT.fits']}
class_weights_all = {'Set1': torch.FloatTensor([0.0323,0.9677]),
                     'Set2': torch.FloatTensor([0.0307,0.9693]),
                     'Set3': torch.FloatTensor([0.0430,0.9570]),
                     'Set4': torch.FloatTensor([0.0602,0.9398]),
                     'Set5': torch.FloatTensor([0.0478,0.9522])}

batch_size = 32
"""
for Set in range(1,6):
   df_final = pd.DataFrame()
   print(f'==================== Set {Set} ====================')

   class_weights = class_weights_all['Set' + str(Set)]
   criterion = nn.CrossEntropyLoss(weight = class_weights)
   to_device(class_weights, device)
   to_device(criterion, device)

   for sav_nm in all_models:
      print('=================== Save name: ', sav_nm)
      num = all_models[sav_nm][Set - 1]
      print('Model number = ', num)
      model = torch.load('/home/nitk233sp001/BestModels_Redone/Set' + str(Set) + '/ResNet18_' + sav_nm + '_' + str(num) + '.pt', map_location=torch.device(device), weights_only=False)
      test_dir = test_dir_list[sav_nm]
      extensions = extensions_list[sav_nm]
   
      testdata = CustDatGen_1ch(test_dir[0],df_lab, extensions[0], transform = None, target_transform = None, rescale = False)
      test_loader = DataLoader(testdata, batch_size = batch_size, shuffle = False)
      test_loader = DeviceDataLoader(test_loader, device)
   
      loss, y, pred = evaluate_model(test_loader, model, criterion, device, disp = False)
      print('Shape of pred: ',pred.shape)
      pred = pred.detach().cpu()
      df_final[sav_nm] = pred.numpy()[:,1]

   df_final['Fold'] = df_lab['Fold']
   df_final['Label'] = df_lab['Label']
   print(df_final)
   df_final.to_csv('/home/nitk233sp001/Ensemble/csv_inps/Set' + str(Set) + '_1chPredEnsemble.csv', header = True, index = False)



"""
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
      #print(f'----------- Set: {Set}')
      #print(f'----------- Model: {num}')
      #print('/home/nitk233sp001/' + model_dirs[sav_nm] + '/Outputs/Set' + str(Set) + '/Pred_csv/ResNet18_' + sav_nm + '_pred_' + str(num)  + '.csv' ) 
      df = pd.read_csv('/home/nitk233sp001/BestModel_Pred_csv/Set' + str(Set) + '/ResNet18_' + sav_nm + '_pred_' + str(num) + '.csv')
      df_all = pd.concat([df_all, df], axis = 0)
      arr = np.concatenate((arr,df_lab[df_lab['Fold'] == Set ]['Label'].values), axis = 0)
      fold = np.concatenate((fold, df_lab[df_lab['Fold'] == Set ]['Fold'].values), axis = 0)
   #print(f'df_all: \n',df_all)
   #print(f'len(arr): ', len(arr))

   df_final[sav_nm] = df_all['P(NoFlare)'].values
   #print(df_final)
   #print('Arr same: ', (arr == arr1).all())

df_final['Label'] = arr.astype('int')
df_final['Fold'] = fold.astype('int')
print(df_final)
df_final.to_csv('/home/nitk233sp001/Ensemble/csv_inps/1chPredEnsemble_noflare.csv', header = True, index = False)

