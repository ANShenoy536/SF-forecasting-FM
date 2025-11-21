#########################
# SF_ResNet18_df_1ch.py
#########################
# This program is for training and dev-testing and imagenet-pretrained ResNet18 model. 
# It gets augmentation specifications and datagenerators from separate python files.'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import torchvision.transforms as v2
#from torchvision.io import read_image, ImageReadMode

from PT_DataGeneratorsFits import CustDatGen_1ch
from augmentationsPt import Train_aug2
from TrainNEval_Pt import *
from PtGPU_helper import *

import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import sys


dir_paths = {'LOS': '../data/LOS_preprocessed', 'USJH': '../data/USJH_preprocessed',
			 'Jz': '../data/Jz_preprocessed', 'USFLUX':  '../data/USFLUX_preprocessed', 'TOTPOT': '../data/TOTPOT_preprocessed']}
all_extensions = {'LOS': '.magnetogram.fits', 'USJH': '.USJH.fits', 'Jz': '.jz.fits',  'USFLUX': '.usflux.fits', 'TOTPOT': '.TOTPOT.fits'}

if len(sys.argv) == 8:
	sav_nm = ing(sys.argv[1])
    Set = int(sys.argv[2])
    num = int(sys.argv[3])
    epochs1 = int(sys.argv[4])
    epochs2 = list(map(int, sys.argv[5].strip('[]').split(',')))
    learning_rate = float(sys.argv[6])
    resnet18_1ch_path = sys.argv[7]
else:
    print('Enter the command line input as: \n\t\tSF_ResNet18_df_1ch.py sav_nm set num epochs1 "epochs2" learning_rate pre_trained_model_path')
    sys.exit(0)
	
device = get_default_device()
print('Device: ',device)

# Give the path for the image directory
train_dir = dir_paths[sav_nm]
test_dir = train_dir
extensions = all_extensions[sav_nm] 

out_dir = os.path.join('../Outputs', sav_nm, 'Set' + Set)
if not os.path.exists(out_dir):
	print('making Outputs directory...')
	os.makedirs(out_dir)
	os.makedirs(os.path.join(out_dir, 'Model'))
	os.makedirs(os.path.join(out_dir, 'Pred_csv'))
	os.makedirs(os.path.join(out_dir, 'ROC'))
	os.makedirs(os.path.join(out_dir, 'LossCurves'))
	
				
f = open(os.path.join(out_dir, 'Results'+ sav_nm + '.csv'),'a')

#Defining the annotation dataframes for training and validation -> 'FileName;Label
anno_df = pd.read_csv('../split_csv/FiveFoldCV.csv')
anno_df_tr = anno_df[anno_df['Fold'] != Set]
anno_df_tr = anno_df_tr.reset_index().drop('index',axis = 1)
anno_df_te = anno_df[anno_df['Fold'] == Set]
anno_df_te = anno_df_te.reset_index().drop('index',axis = 1)

# Define input size and which transforms to use
img_size = 224
transform_noAug = None
transform_Aug = Train_aug2()

# define the custom dataset instance
traindata = CustDatGen_1ch(train_dir,anno_df_tr, extensions, transform = transform_Aug, target_transform = None, rescale = False)
testdata = CustDatGen_1ch(test_dir,anno_df_te, extensions, transform = transform_noAug, target_transform = None, rescale = False)

# load the data using dataloader
batch_size = 32
train_loader = DataLoader(traindata, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(testdata, batch_size = batch_size, shuffle = False)

# load the resnet18 model with the pre-trained weights
resnet18 = torch.load(resnet18_1ch_path, weights_only=False,map_location = device)
layers = [nn.Linear(in_features=512, out_features=2, bias=True )]
resnet18.fc = nn.Sequential(*layers)

print(resnet18)


# Move the dataloader data to gpu using the class
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

'''
print('For training data')
i = 0
for xb, yb in train_loader:
    print('Batch: ', i+1)
    print('xb.device:', xb.device)
    print('yb.device:', yb.device)
    print(xb)
    print('\n\n')
    print(yb)
    i += 1

print('For testing data')
i=0
for xb, yb in test_loader:
    print('Batch: ', i+1)
    print('xb.device:', xb.device)
    print('yb.device: ', yb.device)
    i += 1
'''

# Move the model to the gpu
to_device(resnet18, device)

'''
for params in resnet18.parameters():
    print('Param.device: ',params.device)
'''

# Defining hyperparameters for training
hist_all = []
# Class weights for the five folds
all_class_weights = {1: torch.FloatTensor([0.0323,0.9677]), 2: torch.FloatTensor([0.0307,0.9693]), 3: torch.FloatTensor([0.0430,0.9570]), 4: torch.FloatTensor([0.0602,0.9398]), 5: torch.FloatTensor([0.0478,0.9522])}
class_weights = all_class_weights[Set]

# Defining the loss function
criterion = nn.CrossEntropyLoss(weight = class_weights)

to_device(class_weights, device)
to_device(criterion, device)

# Defining the optimizer
optimizer = torch.optim.Adam(resnet18.parameters(), lr = learning_rate)

#----------------Training only the last (FC) layer -------------------
print('Training only the last layer')

for name, param in resnet18.named_parameters():
  if (name.startswith('fc')):
    param.requires_grad = True
  else:
    param.requires_grad = False

resnet18, hist1 = fit_model(resnet18, epochs1, learning_rate, criterion, optimizer, train_loader)

# ----------------Training the entire network-------------------------
print('\nTraining the entire network')

for name, param in resnet18.named_parameters():
  param.requires_grad = True

for i in range(len(epochs2)):
	resnet18, hist2 = fit_model(resnet18, epochs2[i], learning_rate, criterion, optimizer, train_loader)


	# Saving the model
	torch.save(resnet18, os.path.join(out_dir, "Model/ResNet18_"+sav_nm + "_" + str(num) + ".pt"))

	# Do inference of the testing data

	# Getting training metrics
	#print('Evaluating Training data')
	train_loss_bw, y_train_all, pred_train_all = evaluate_model(train_loader, resnet18, criterion, device, disp = False)

	y_train_all = y_train_all.cpu()
	pred_train_all = pred_train_all.cpu()
	train_cm = confusion_matrix(y_train_all, torch.argmax(pred_train_all, dim = 1))
	train_acc = (train_cm[0,0]+train_cm[1,1])/(train_cm[0,0]+train_cm[1,1]+train_cm[0,1]+train_cm[1,0])

	train_loss_all = torch.mean(train_loss_bw).item()

	print(f'\nTraining loss: {train_loss_all}')
	print(f'Training accuracy: {train_acc} \n')

	# Getting testing metrics
	#print('Evaluating Testing data')
	test_loss_bw, y_test_all, y_pred_all = evaluate_model(test_loader, resnet18, criterion, device, disp = False)

	y_test_all = y_test_all.detach().cpu()
	y_pred_all = y_pred_all.detach().cpu()

	#saving the validation predictions in a csv file
	df = pd.DataFrame(y_pred_all.numpy(), columns = ['P(NoFlare)', 'P(Flare)'])
	df.to_csv(os.path.join(out_dir, 'Pred_csv/ResNet18_' + sav_nm + '_pred_' + str(num) + '.csv'), header = True, index = False)


	'''
	print(y_test_all.device)
	print(y_pred_all.device)
	print('y_test_all: \n', y_test_all)
	print('y_pred_all: \n', y_pred_all)
	'''

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
	plt.title('AUC-ROC curve of testing data')
	plt.savefig(os.path.join(out_dir, 'ROC/ResNet18_' + sav_nm + '_roc_' + str(num) + '.png'))

	if len(hist2):
		hist_all = hist_all + hist2
	plt.figure()
	plt.plot(hist1, c = 'r', label = 'Last 2 layers')

	if len(hist_all):
		plt.plot(hist_all, c = 'k', label = 'Entire network')
	plt.title('Training loss curve')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(os.path.join(out_dir, 'LossCurves/ResNet18_' + sav_nm + '_loss_' + str(num) + '.png'))
	
	print('---------------------->num = ', num)

	f.write(str(Set) + ',' + str(num) + ',' + sav_nm + ',' + str(learning_rate) + ',' + str(cm[1,1]) + ',' + str(cm[0,0]) + ',' + str(cm[0,1]) + ',' + str(cm[1,0]) + ',' + str(acc_c) + ',' + str(auc) + ',' + str(prec_c) + ',' + str(rec_c) + ',' + str(f1_c) + ',' + str(tss) + '\n')
        
	num = num + 1
f.close()

