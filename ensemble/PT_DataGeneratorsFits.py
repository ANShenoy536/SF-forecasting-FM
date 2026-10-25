''' This file contains the custom data generator for 1, 2 and 3 channels '''

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import os

from astropy.io import fits



# Custom dataset for ensemble model
class CustDatGen_Ensemble(torch.utils.data.Dataset):
    def __init__(self, in_len, anno_df, transform = None, target_transform = None, rescale = False):
        self.in_len = in_len
        self.anno_df = anno_df
        self.transform= transform
        self.target_transform = target_transform
        self.rescale = rescale

        self.col_max = np.max(anno_df.iloc[:,:self.in_len], axis = 0)
        self.n = len(self.anno_df)
        self.n_classes = self.anno_df['Label'].nunique()
        print(f'{self.n}  files found belonging to {self.n_classes} classes')


    def __len__(self):
        return self.n



    def __getitem__(self, idx):
        x = self.anno_df.iloc[idx,:self.in_len]
        y = self.anno_df['Label'].values[idx]

        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        if self.rescale:
           x = x/self.col_max

        if self.transform:
           x = self.transform(x)

        if self.target_transform:
           y = self.target_transform(y)

        return x, y
