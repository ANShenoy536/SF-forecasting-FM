"""
This script generates predeictions i.e. logit(Flare) for the entire dataset using the best models from each set, for all 5 individual FMs. For every set, it stores the logits from the best model for each of the 5 FMs. Therefore, the output is 5 csv files, one for each set, with header being the different FMs.
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


df_lab = pd.read_csv('../split_csv/FiveFoldCV.csv')

all_models = {'sav_nm': list of model_num for fold-wise best model}
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
      model = torch.load('../Outputs/' + sav_nm + '/Set' + str(Set) + '/Model/ResNet18_' + sav_nm + '_' + str(num) + '.pt', map_location=torch.device(device), weights_only=False)
      test_dir = test_dir_list[sav_nm]
      extensions = extensions_list[sav_nm]
   
      testdata = CustDatGen_1ch(test_dir[0],df_lab, extensions[0], transform = None, target_transform = None, rescale = False)
      test_loader = DataLoader(testdata, batch_size = batch_size, shuffle = False)
      test_loader = DeviceDataLoader(test_loader, device)
   
      loss, y, pred = evaluate_model(test_loader, model, criterion, device, disp = False)
      print('Shape of pred: ',pred.shape)
      pred = pred.detach().cpu()
      df_final[sav_nm] = pred.numpy()[:,1] # getting the logit(Flare) only

   df_final['Fold'] = df_lab['Fold']
   df_final['Label'] = df_lab['Label']
   print(df_final)
   df_final.to_csv('csv_inps/Set' + str(Set) + '_1chPredEnsemble.csv', header = True, index = False)

