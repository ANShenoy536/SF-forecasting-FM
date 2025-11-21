#######################
# GradCAM_Pt_5fold.py
#######################
# Generating the Grad-CAM heatmaps for the test-fold in each of the 5 fold CV sets

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import resnet18, resnet50
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pandas import read_csv
import os
import cv2
from PtGPU_helper import *
import sys

print('Starting the codels in the file')

class ResNet(nn.Module):
    def __init__(self, model_path, device):
        super(ResNet, self).__init__()

        self.model_path = model_path
        # get the pretrained VGG19 network
        self.resnet = torch.load(model_path, map_location=device) 

        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(self.resnet.conv1,
                                            self.resnet.bn1,
                                            self.resnet.relu,
                                            self.resnet.maxpool,
                                            self.resnet.layer1,
                                            self.resnet.layer2,
                                            self.resnet.layer3,
                                            self.resnet.layer4)
        # get the max pool of the features stem
        self.remainintB4classifier = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # get the classifier of the vgg19
        self.classifier = self.resnet.fc[0]

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # print(x.shape)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.remainintB4classifier(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)



def read_img(file_name, dirs, extensions):
  if len(dirs) == 3:
      img_path1 = os.path.join(dirs[0], file_name) + extensions[0]
      img_path2 = os.path.join(dirs[1], file_name) + extensions[1]
      img_path3 = os.path.join(dirs[2], file_name) + extensions[2]

      img1 = np.expand_dims(fits.getdata(img_path1), axis = 0)
      img2 = np.expand_dims(fits.getdata(img_path2), axis = 0)
      img3 = np.expand_dims(fits.getdata(img_path3), axis = 0)

      img = np.concatenate([img1, img2, img3], axis = 0)
      img = np.expand_dims(img, axis = 0)
      img = torch.from_numpy(img)
      img = img.type(torch.FloatTensor)
      img_list = [img, img1, img2, img3]
  elif len(dirs) == 2:
      img_path1 = os.path.join(dirs[0], file_name) + extensions[0]
      img_path2 = os.path.join(dirs[1], file_name) + extensions[1]

      img1 = np.expand_dims(fits.getdata(img_path1), axis = 0)
      img2 = np.expand_dims(fits.getdata(img_path2), axis = 0)

      img = np.concatenate([img1, img2], axis = 0)
      img = np.expand_dims(img, axis = 0)
      img = torch.from_numpy(img)
      img = img.type(torch.FloatTensor)
      img_list = [img, img1, img2]
  else:
      img_path1 = os.path.join(dirs[0], file_name) + extensions[0]

      img1 = np.expand_dims(fits.getdata(img_path1), axis = 0)

      img = np.concatenate([img1], axis = 0)
      img = np.expand_dims(img, axis = 0)
      img = torch.from_numpy(img)
      img = img.type(torch.FloatTensor)
      img_list = [img, img1]

  return img_list



def heatmap_lastlayer(pred, resnet):
  pred[:,1].backward()

  # pull the gradients out of the model
  gradients = resnet.get_activations_gradient()
  #print('Shape of gradients: ', gradients.shape)

  # pool the gradients across the channels
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  #print('Shape of pooled gradient: ', pooled_gradients.shape)

  # get the activations of the last convolutional layer
  activations = resnet.get_activations(img).detach()

  # weight the channels by corresponding gradients
  ch = 512 # for resnet18
  #ch = 2048 # for resnet50
  for i in range(ch):
      activations[:, i, :, :] *= pooled_gradients[i]

  #print('Shape of activations: ', activations.shape)
  # average the channels of the activations
  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = heatmap.cpu().numpy()
  #print('Shape of heatmap: ', heatmap.shape)
  #print('Unique values in heatmap: \n', np.unique(heatmap))
  # print(heatmap)

  # relu on top of the heatmap
  heatmap = np.maximum(heatmap, 0)

  # normalize the heatmap
  if np.max(heatmap):
      heatmap /= np.max(heatmap)
  return heatmap


def save_image1(path, img1, heatmap, label, pred, Set, cm_cls, names):
  plt.figure(figsize = (9,9))
  plt.suptitle(f'Set = {Set}\nPrediction = {pred_cls}   GT = {label}  CM_cls = {cm_cls}')
  plt.subplots_adjust(hspace = 0.2)
  plt.subplot(121)
  plt.title(names[0], fontsize=10)
  plt.imshow(img1[0], cmap = 'gray')
  
  plt.subplot(122)
  heatmap2 = cv2.resize(heatmap, (224,224), interpolation = cv2.INTER_CUBIC)
  plt.imshow(heatmap2)
  #plt.colorbar()
  plt.savefig(path)
  plt.close()

 

def save_image3(path, img1, img2, img3, heatmap, label, pred, Set, cm_cls, names):
  plt.figure(figsize = (9,9))
  plt.suptitle(f'Set = {Set}\nPrediction = {pred_cls}   GT = {label}  CM_cls = {cm_cls}')
  plt.subplots_adjust(hspace = 0.2)
  plt.subplot(221)
  plt.title(names[0], fontsize=10) #') #Bp')
  plt.imshow(img1[0], cmap = 'gray')
  plt.subplot(222)
  plt.title(names[1], fontsize=10) #'USFLUX') #Br')
  plt.imshow(img2[0], cmap = 'gray')
  plt.subplot(223)
  plt.title(names[2], fontsize=10) #Bt') #'TOTPOT')
  plt.imshow(img2[0], cmap = 'gray')
  plt.subplot(224)
  heatmap2 = cv2.resize(heatmap, (224,224), interpolation = cv2.INTER_CUBIC)
  plt.imshow(heatmap2)
  #plt.colorbar()
  plt.savefig(path)
  plt.close()



def save_image2(path, img1, img2, heatmap, label, pred_cls,Set, cm_cls,names):
  plt.figure(figsize = (9,4))
  plt.suptitle(f'Set = {Set}\nPrediction = {pred_cls}   GT = {label}  CM_cls = {cm_cls}')
  plt.subplots_adjust(hspace = 0.2)
  plt.subplot(131)
  plt.title(names[0],fontsize=10) #Bp')
  plt.imshow(img1[0], cmap = 'gray')
  plt.subplot(132)
  plt.title(names[1],fontsize=10) #Br')
  plt.imshow(img2[0], cmap = 'gray')
  plt.subplot(133)
  heatmap2 = cv2.resize(heatmap, (224,224), interpolation = cv2.INTER_CUBIC)
  plt.imshow(heatmap2)
  #plt.colorbar()
  plt.savefig(path)
  plt.close()

def get_cm_cls(pred_cls, label):
  if label == 1:
     if pred_cls == 1:
        return 'TP'
     elif pred_cls == 0:
        return 'FN'
  elif label == 0:
     if pred_cls == 1:
        return 'FP'
     elif pred_cls == 0:
        return 'TN'

def save_img_fits(heatmap, sav_pth):
    img = cv2.resize(heatmap, (224,224), interpolation = cv2.INTER_CUBIC)
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(sav_pth, overwrite=True)


if len(sys.argv) == 3:
   sav_nm = sys.argv[1]
   models = list(map(int, sys.argv[2].strip('[]').split(',')))
else:
   print('Enter the command line input as: \n\t\tGradCAM_Pt_5fold.py sav_nm "models"')
   sys.exit(0)

all_dirs = {'Vmgm': ['../data/Bp_preprocessed', '../data/Br_preprocessed', '../data/Bt_preprocessed'],
'JFP': ['../data/Jz_preprocessed', '../data/USFLUX_preprocessed', '../data/TOTPOT_preprocessed'],
'FP': ['../data/USFLUX_preprocessed', '../data/TOTPOT_preprocessed'],
'Jz': ['../data/Jz_preprocessed'],
'USJH': ['../data/USJH_preprocessed'],
'LOS': ['../data/LOS_preprocessed'],
'USFLUX': ['../data/USFLUX_preprocessed'],
'TOTPOT': ['../data/TOTPOT_preprocessed']}

all_exts = {'Vmgm': ['.Bp.fits', '.Br.fits', '.Bt.fits'],
'JFP': ['.jz.fits', '.usflux.fits', '.TOTPOT.fits'],
'FP': ['.usflux.fits', '.TOTPOT.fits'],
'Jz': ['.jz.fits'],
'USJH': ['.USJH.fits'],
'LOS': ['.magnetogram.fits'],
'USFLUX': ['.usflux.fits'],
'TOTPOT': ['.TOTPOT.fits']}

all_names = { 'Vmgm': ['$B_\phi$', '$B_r$', '$B_\\theta$'], 
'JFP': ['Vertical current', 'Unsigned magnetic flux', 'Magnetic energy density'],
'FP': ['Unsigned magnetic flux', 'Magnetic energy density'],
'Jz': ['Vertical current'],
'USJH': ['Unsigned current helicity'],
'LOS': ['LoS magnetogram'],
'USFLUX': ['Unsigned magnetic flux'],
'TOTPOT': ['Magnetic energy density']}

dirs = all_dirs[sav_nm]
extensions = all_exts[sav_nm]
names = all_names[sav_nm]

csv_path = '../split_csv/FiveFoldCV.csv'
df = read_csv(csv_path)

print(f'------------------------------------------------------------------------------------------')
print(f'--------------------------- Data: {sav_nm} ----------------------------') 


#print('Checkpoint 2')
device = get_default_device()
print('Device: ', device)

for k in range(len(models)):
    num = models[k]
    Set = k+1
    df_test = df[df['Fold'] == Set]
    df_test = df_test.reset_index().drop('index',axis = 1)
    
    dest = os.path.join('ReHeatMaps_5fold', sav_nm, 'Set'+str(Set))
    if not os.path.exists(dest):
        os.makedirs(dest)
    model_path = os.path.join('../Outputs', sav_nm, 'Set' + str(Set), "Model/ResNet18_"+sav_nm + "_" + str(num) + ".pt")
                              
    resnet = ResNet(model_path, device)
    resnet.eval()
#print('Checkpoint 3: Model loaded and put in eval mode')
    print(f'=============== Testing Fold {Set} for Set {Set} ==============')
    for i in range(len(df_test)):
       fl_name = df_test['FileName'].values[i]
       label = df_test['Label'].values[i]
       img_list = read_img(fl_name, dirs, extensions)
       img = img_list[0]
       img = img.to(device)
        
       pred = resnet(img)
       heatmap = heatmap_lastlayer(pred, resnet)

       pred_cls = np.argmax(pred.detach().cpu().numpy(), axis = 1)[0]
       cm_cls = get_cm_cls(pred_cls, label)
       #print(f'label = {label}   pred_cls = {pred_cls}   cm_cls = {cm_cls}')

       #check if cm folder exists. If not make it
       if not os.path.exists(os.path.join(dest,cm_cls + '/')):
          os.makedirs(os.path.join(dest,cm_cls + '/'), exist_ok=True)
       sav_path = os.path.join(dest, cm_cls + '/' + fl_name) + '.heatmap.png'

       save_img_fits(heatmap, sav_path)
 
       #print(f'Completed: {fl_name}')
