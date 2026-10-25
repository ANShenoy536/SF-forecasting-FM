import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PT_DataGeneratorsFits import *
from augmentationsPt import Train_aug2
from TrainNEval_Pt import *
from EnsembleModelClass import EnsembleModel
from PtGPU_helper import *

import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 5:
   Set = int(sys.argv[1])
   num = int(sys.argv[2])
   epochs = int(sys.argv[3])
   learning_rate = float(sys.argv[4])
else:
   print('Please enter command line input as \n \t\t SF_Ensemble.py Set num epochs learning_rate')
   sys.exit(0)


device = get_default_device()
print('Device: ', device)

print(f'================== Set {Set} =====================')

anno_df = pd.read_csv('/home/nitk233sp001/Ensemble/csv_inps/Set' + str(Set) + '_1chPredEnsemble.csv')
anno_df_tr = anno_df[anno_df['Fold'] != Set]
anno_df_tr = anno_df_tr.reset_index().drop('index',axis = 1)
anno_df_te = anno_df[anno_df['Fold'] == Set]
anno_df_te = anno_df_te.reset_index().drop('index',axis = 1)

img_size = 224
batch_size = 32
transform_noAug = None
transform_Aug = None #Train_aug2()

# --- DataLoaders ---
en_tr = CustDatGen_Ensemble(5, anno_df_tr, transform = transform_Aug, target_transform = None, rescale = False)
en_te = CustDatGen_Ensemble(5, anno_df_te, transform = transform_noAug, target_transform = None, rescale = False)

traindata = DataLoader(en_tr, batch_size = batch_size, shuffle = True)
testdata = DataLoader(en_te, batch_size = batch_size, shuffle = False)

#--- Loading the DLs onto the device ---
train_loader = DeviceDataLoader(traindata, device)
test_loader = DeviceDataLoader(testdata, device)

class_weights_all = {'Set1': torch.FloatTensor([0.0323,0.9677]), 
                     'Set2': torch.FloatTensor([0.0307,0.9693]), 
                     'Set3': torch.FloatTensor([0.0430,0.9570]), 
                     'Set4': torch.FloatTensor([0.0602,0.9398]),
                     'Set5': torch.FloatTensor([0.0478,0.9522])}

model = EnsembleModel()
class_weights = class_weights_all['Set' + str(Set)]
criterion = nn.CrossEntropyLoss(weight = class_weights)

to_device(class_weights, device)
to_device(criterion, device)
to_device(model, device)
print(model)

# --- Defining hyperparameters for training ---
save_dest = '/home/nitk233sp001/Ensemble/Outputs/Model/ResNet18_Ens' + str(Set) + '_' + str(num) + '.pt'
pred_dest = '/home/nitk233sp001/Ensemble/Outputs/Pred_csv/ResNet18_Ens' + str(Set) + '_pred_' + str(num) + '.csv'
roc_dest = '/home/nitk233sp001/Ensemble/Outputs/ROC/ResNet18_Ens' + str(Set) + '_roc_' + str(num) + '.png'
loss_dest = '/home/nitk233sp001/Ensemble/Outputs/FiveFoldCV/LossCurves/ResNet18_Ens' + str(Set) + '_loss_' + str(num) + '.png'

hist_all = []

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for name, param in model.named_parameters():
  param.requires_grad = True

model,hist = fit_model(model, epochs, learning_rate, criterion, optimizer, train_loader)
torch.save(model,save_dest)

# --- Training metrics ---
train_loss_bw, y_train_all, pred_train_all = evaluate_model(train_loader, model, criterion, device, disp = False)

y_train_all = y_train_all.cpu()
pred_train_all = pred_train_all.cpu()
train_cm = confusion_matrix(y_train_all, torch.argmax(pred_train_all, dim = 1))
train_acc = (train_cm[0,0]+train_cm[1,1])/(train_cm[0,0]+train_cm[1,1]+train_cm[0,1]+train_cm[1,0])

train_loss_all = torch.mean(train_loss_bw).item()

print(f'\nTraining loss: {train_loss_all}')
print(f'Training accuracy: {train_acc} \n')

# --- Testing metrics ---
test_loss_bw, y_test_all, y_pred_all = evaluate_model(test_loader, model, criterion, device, disp = False)

y_test_all = y_test_all.detach().cpu()
y_pred_all = y_pred_all.detach().cpu()

# --- saving the validation predictions in a csv file ---
df = pd.DataFrame(y_pred_all.numpy(), columns = ['P(NoFlare)', 'P(Flare)'])
df.to_csv(pred_dest, header = True, index = False)

print('Calculated Test results: ')
cm = confusion_matrix(y_test_all, torch.argmax(y_pred_all, dim = 1))
print('Confusion matrix: ', cm)

test_loss_all = torch.mean(test_loss_bw).item()
acc_c = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
prec_c = cm[1,1]/(cm[1,1]+cm[0,1])
rec_c = cm[1,1]/(cm[1,1]+cm[1,0])
f1_c = (2* prec_c * rec_c)/(prec_c + rec_c)
tss = rec_c - cm[0,1]/(cm[0,0] + cm[0,1])

auc = roc_auc_score(y_test_all, y_pred_all[:,1])

print('Test loss: ', test_loss_all)
print('Accuracy: ', acc_c)
print('AUC (testing): ', auc)
print('Precision: ', prec_c)
print('Recall: ', rec_c)
print('F1-score: ', f1_c)
print('TSS: ', tss)

fpr, tpr, th = roc_curve(y_test_all, y_pred_all[:,1])

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.title('AUC-ROC curve of testing data - with CW')
plt.savefig(roc_dest)
                
plt.figure()
plt.plot(hist, c = 'r', label = 'Trainging loss')
plt.title('Training loss curve - with CW')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_dest)

print('-----------------------------> num = ', num)
