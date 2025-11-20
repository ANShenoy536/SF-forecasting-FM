import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as v2
from torchvision.io import read_image, ImageReadMode

from PT_DataGeneratorsFits import *
from PtGPU_helper import *

# from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

"""
fl = open('/home/nitk233sp001/Eval/Results_csv/TestResults/ResultsFitsPtCV_ext_FP.csv','a')
fl.write('Model#;FM;Split;Threshold;Loss;TN;FP;FN;TP;Accuracy;AUC;Precision;Recall;Fscore;TSS\n')
fl.close()
"""

# ----------------Function to generate the prediction labels----------------#
def pred_labels(y_pred, th):
    y_pred2 = y_pred[:,1]
    pred_lab = np.zeros_like(y_pred2)
    pred_lab[y_pred2 >= th] = 1
    return pred_lab


# -----------------------------Eval function -----------------------------#

def evaluate_model(test_loader, model, criterion, disp = True):
    test_loss_bw = [] # batchwise test loss
    y_test_all = torch.tensor([], device = device)
    y_pred_all = torch.tensor([], device = device)

    model.eval()
    with torch.no_grad():
        for te_batch, (X_test, y_test) in enumerate(test_loader):
            # get the model prediction (score)
            y_pred_test = model(X_test)

            # compute the loss
            loss = criterion(y_pred_test, y_test)

            # append this loss to the batch_wise test loss
            test_loss_bw.append(loss.item())

            # append the y_test to the y_test_all
            y_test_all = torch.cat((y_test_all, y_test), dim = 0)

            # append the y_pred to y_pred_all list
            y_pred_all = torch.cat((y_pred_all, y_pred_test), dim = 0)

            #Print some message for each batch
            if disp:
               print(f'Testing batch {te_batch + 1}/{len(test_loader)}')
    return torch.tensor(test_loss_bw, device = device), y_test_all, y_pred_all

# ----------------------Getting the testing data ---------------------------#

device = get_default_device()
print('Device: ',device)

#num = 6
BS = 32
img_size = 224

transform_noAug = None #v2.Compose([v2.Resize(size = (img_size, img_size))])

# For VectMgm
test_dir = ['/home/nitk233sp001/RawFM_fits/Bp2', '/home/nitk233sp001/RawFM_fits/Br2', '/home/nitk233sp001/RawFM_fits/Bt2']
extensions = ['.Bp.fits', '.Br.fits', '.Bt.fits']

#For JzFluxPot
#test_dir = ['/home/nitk233sp001/RawFM_fits/Jz2', '/home/nitk233sp001/RawFM_fits/USFLUX2', '/home/nitk233sp001/RawFM_fits/TOTPOT2']
#extensions =  ['.jz.fits', '.usflux.fits', '.TOTPOT.fits']

#For FluxPot
#test_dir = ['/home/nitk233sp001/RawFM_fits/USFLUX2', '/home/nitk233sp001/RawFM_fits/TOTPOT2']
#extensions = ['.usflux.fits', '.TOTPOT.fits']

#For Single channel models
#test_dir = '/home/nitk233sp001/RawFM_fits/Jz2'
#test_dir = '/home/nitk233sp001/RawFM_fits/USFLUX2'
#test_dir = '/home/nitk233sp001/RawFM_fits/LOS2'
#test_dir = '/home/nitk233sp001/RawFM_fits/USJH2'
#test_dir = '/home/nitk233sp001/RawFM_fits/TOTPOT2'

#extensions =  '.jz.fits'
#extensions =  '.usflux.fits'
#extensions = '.magnetogram.fits'
#extensions = '.USJH.fits'
#extensions =  '.TOTPOT.fits'

sav_nm = 'Vmgm'
arch_nm = 'ResNet18' #'Vgg16'
#Split = 1
model_numbers = [24,34,9,3,14]
threshold = 0.5
anno_df_te = pd.read_csv('/home/nitk233sp001/Preprocessing/70_20_10/FiveFoldCV.csv') 
#pd.read_csv('/home/nitk233sp001/Preprocessing/70_20_10/Split' + str(Split) + '_extended.csv') #, sep = ';')
#anno_df_te = anno_df[anno_df['Split'] == 'Test']
#anno_df_te = anno_df_te.iloc[:,0:2]

class_weights_dict = {'set1': torch.FloatTensor([0.0323,0.9677]), 'set2': torch.FloatTensor([0.0307,0.9693]), 'set3': torch.FloatTensor([0.0430,0.9570]), 'set4': torch.FloatTensor([0.0602,0.9398]), 'set5': torch.FloatTensor([0.0478,0.9522])}

testdata = CustDatGen_3ch(test_dir[0], test_dir[1],test_dir[2], anno_df_te, extensions, transform = transform_noAug, target_transform = None, rescale = False)
#testdata = CustDatGen_2ch(test_dir[0], test_dir[1], anno_df_te, extensions, transform = transform_noAug, target_transform = None, rescale = False)
#testdata = CustDatGen_1ch(test_dir, anno_df_te, extensions, transform = transform_noAug, target_transform = None, rescale = False)

test_loader = DataLoader(testdata, batch_size = BS, shuffle = False)
test_loader = DeviceDataLoader(test_loader, device)


for i in range(len(model_numbers)):
    num  = model_numbers[i]
    Split = i + 1
    class_weights = class_weights_dict['set' + str(Split)]
    print('\n\nTESTING MODEL-->',num,'\n')
  
    # ---------------------------Loading the model ------------------------------#

    model = torch.load('/home/nitk233sp001/VectMgm/Outputs/Set'+ str(Split) + '/Model/' + arch_nm + '_' + sav_nm + '_' + str(num) + '.pt', map_location=device)
    #model = torch.load('/scratch/nitk233sp001/VectMgm/Outputs/NewAug_Split2/Model/' + arch_nm + '_' + sav_nm + '_' + str(num) + '.pt', map_location=device)
    to_device(model, device)
    
    #class_weights = torch.FloatTensor([0.0415,0.9585]) # Split 1
    #class_weights = torch.FloatTensor([0.0407,0.9593]) # Split2
    #class_weights = torch.FloatTensor([0.0422,0.9578]) # Split3
    #class_weights = torch.FloatTensor([0.0428,0.9572]) # Split4
    #class_weights = torch.FloatTensor([0.0452,0.9548]) # Split5 
    #class_weights = torch.FloatTensor([0.0397,0.9603]) # Split6


    #class_weights = torch.FloatTensor([0.0323,0.9677]) # Set1
    #class_weights = torch.FloatTensor([0.0307,0.9693]) # Set2
    #class_weights = torch.FloatTensor([0.0430,0.9570]) # Set3
    #class_weights = torch.FloatTensor([0.0602,0.9398]) # Set4
    #class_weights = torch.FloatTensor([0.0478,0.9522]) # Set5


    criterion = nn.CrossEntropyLoss(weight = class_weights)
    #criterion = nn.CrossEntropyLoss() #for the manually balanced case
    
    to_device(class_weights, device)
    to_device(criterion, device)
    # --------------------------------Testing -----------------------------------#

    # Evaluating the model to show the loss and metrics 
    # Getting testing metrics
    #print('Evaluating Testing data')
    test_loss_bw, y_test_all, y_pred_all = evaluate_model(test_loader, model, criterion, disp = False)

    y_test_all = y_test_all.detach().cpu()   # all the test ground truth labels
    y_pred_all = y_pred_all.detach().cpu()   # all the test softmax outputs
    
    y_pred_lab = pred_labels(y_pred_all, threshold)   # all the prediction labels

    #saving the validation predictions in a csv file
    df = pd.DataFrame(y_pred_all.numpy(), columns = ['P(NoFlare)', 'P(Flare)'])
    df.to_csv('/home/nitk233sp001/Eval/FiveFold/Pred_csv/Set' + str(Split) + '/' + arch_nm + '_' + sav_nm + '_pred_' + str(num) + '.csv', header = True, index = False)
   
    print('Calculated Test results: ')
    cm = confusion_matrix(y_test_all, y_pred_lab)
    print('Confusion matrix: ', cm)

    test_loss_all = torch.mean(test_loss_bw).item()
    acc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
    prec = cm[1,1]/(cm[1,1]+cm[0,1])
    rec = cm[1,1]/(cm[1,1]+cm[1,0])
    f1 = (2* prec * rec)/(prec + rec)
    tss = rec - cm[0,1]/(cm[0,0] + cm[0,1])

    auc_test = roc_auc_score(y_test_all, y_pred_all[:,1])

    print('Test loss: ', test_loss_all)
    print('Accuracy: ', acc)
    print('AUC (testing): ', auc_test)
    print('Precision: ', prec)
    print('Recall: ', rec)
    print('F1-score: ', f1)
    print('TSS: ', tss)

    fpr, tpr, th = roc_curve(y_test_all, y_pred_all[:,1])

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR (Recall)')
    plt.title('AUC-ROC curve of testing data')
    plt.savefig('/home/nitk233sp001/Eval/FiveFold/ROC/Set' + str(Split) + '/' + arch_nm + '_' + sav_nm + '_roc_' + str(num) + '.png')

"""
    fl = open('/home/nitk233sp001/Eval/Results_csv/TestResults/NewerAug/ResultsFitsPtCV_ext_Vmgm.csv','a')
    fl.write(arch_nm+ '_' + str(num) + ';' + sav_nm + ';' + str(Split) + ';' + str(threshold) + ';' + str(test_loss_all) + ';' + str(cm[0,0]) + ';' + str(cm[0,1]) + ';' + str(cm[1,0]) + ';' + str(cm[1,1]) + ';' + str(acc) + ';' + str(auc_test) + ';' + str(prec) + ';' + str(rec) + ';' + str(f1) + ';' + str(tss) + '\n')
    fl.close()
"""
