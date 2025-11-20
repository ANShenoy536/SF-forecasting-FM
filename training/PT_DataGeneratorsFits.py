''' This file contains the custom data generator for 1, 2 and 3 channels '''

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import os

from astropy.io import fits


#Custom dataset generator for 3 channel
class CustDatGen_3ch(torch.utils.data.Dataset):
    def __init__(self, dir1, dir2, dir3, anno_df, extensions, transform = None, target_transform = None, rescale = False):
        #self.img_labels = pd.read_csv(annotations_file, sep = ';')
        self.dir1 = dir1
        self.dir2 = dir2
        self.dir3 = dir3
        self.anno_df = anno_df
        self.extensions = extensions
        self.transform = transform
        self.target_transfrom = target_transform
        self.rescale = rescale

        self.n = len(anno_df)
        self.n_classes = anno_df['Label'].nunique()
        print(f'{self.n} files found belonging to {self.n_classes}')

    def __len__(self):
        return self.n
 
   
    def __getitem__(self, idx):
        img_path1 = os.path.join(self.dir1, self.anno_df['FileName'].values[idx]) + self.extensions[0]
        img_path2 = os.path.join(self.dir2, self.anno_df['FileName'].values[idx]) + self.extensions[1]
        img_path3 = os.path.join(self.dir3, self.anno_df['FileName'].values[idx]) + self.extensions[2]

        img1 = np.expand_dims(fits.getdata(img_path1), axis = 0)#read_image(img_path1, mode = ImageReadMode.GRAY) # read_image returns the image in (#channles, image_size) format
        img2 = np.expand_dims(fits.getdata(img_path2), axis = 0)#read_image(img_path2, mode=ImageReadMode.GRAY)
        img3 = np.expand_dims(fits.getdata(img_path3), axis = 0)#read_image(img_path3, mode=ImageReadMode.GRAY)
        image_np = np.concatenate((img1,img2, img3), axis = 0)

        image = torch.from_numpy(image_np)
        image = image.type(torch.FloatTensor)
        label = self.anno_df['Label'].values[idx]

        if self.rescale:
            image = image * (1./255)
        if self.transform:
            image = self.transform(image)
        if self.target_transfrom:
            label = self.target_transfrom(label)

        return image, label


# Custom dataset generator for 2 channels
class CustDatGen_2ch(torch.utils.data.Dataset):
    def __init__(self, dir1, dir2, anno_df, extensions, transform = None, target_transform = None, rescale = False):
        #self.img_labels = pd.read_csv(annotations_file, sep = ';')
        self.dir1 = dir1
        self.dir2 = dir2
    
        self.anno_df = anno_df
        self.extensions = extensions
        self.transform = transform
        self.target_transfrom = target_transform
        self.rescale = rescale

        self.n = len(anno_df)
        self.n_classes = anno_df['Label'].nunique()
        print(f'{self.n} files found belonging to {self.n_classes}')

    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        img_path1 = os.path.join(self.dir1, self.anno_df['FileName'].values[idx]) + self.extensions[0]
        img_path2 = os.path.join(self.dir2, self.anno_df['FileName'].values[idx]) + self.extensions[1]

        img1 = np.expand_dims(fits.getdata(img_path1), axis = 0)#read_image(img_path1, mode = ImageReadMode.GRAY) # read_image returns the image in (#channles, image_size) format
        img2 = np.expand_dims(fits.getdata(img_path2), axis = 0)#read_image(img_path2, mode=ImageReadMode.GRAY)
        image_np = np.concatenate((img1,img2), axis = 0)

        image = torch.from_numpy(image_np)
        image = image.type(torch.FloatTensor)
        label = self.anno_df['Label'].values[idx]

        if self.rescale:
            image = image * (1./255)
        if self.transform:
            image = self.transform(image)
        if self.target_transfrom:
            label = self.target_transfrom(label)

        return image, label


# Custom dataset generator for 1 channel
class CustDatGen_1ch(torch.utils.data.Dataset):
    def __init__(self, dir1, anno_df, extensions, transform = None, target_transform = None, rescale = False):
        #self.img_labels = pd.read_csv(annotations_file, sep = ';')
        self.dir1 = dir1
        
        self.anno_df = anno_df
        self.extensions = extensions
        self.transform = transform
        self.target_transfrom = target_transform
        self.rescale = rescale

        self.n = len(anno_df)
        self.n_classes = anno_df['Label'].nunique()
        print(f'{self.n} files found belonging to {self.n_classes}')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path1 = os.path.join(self.dir1, self.anno_df['FileName'].values[idx]) + self.extensions
        image_np = np.expand_dims(fits.getdata(img_path1), axis = 0)#read_image(img_path1, mode = ImageReadMode.GRAY) # read_image returns the image in (#channles, image_size) format
        #image_np = np.concatenate((img1,img2, img3), axis = 0)

        image = torch.from_numpy(image_np)
        image = image.type(torch.FloatTensor)
        label = self.anno_df['Label'].values[idx]

        if self.rescale:
            image = image * (1./255)
        if self.transform:
            image = self.transform(image)
        if self.target_transfrom:
            label = self.target_transfrom(label)

        return image, label
