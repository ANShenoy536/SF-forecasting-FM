import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch

df_flare = pd.read_csv('csv_inps/1chPredEnsemble_Flare.csv')
df_noflare = pd.read_csv('csv_inps/1chPredEnsemble_NoFlare.csv')

labels = df_flare['Label'].values

print('\nJz, USFLUX, LOS, USJH, TOTPOT\n')

avg_in_flare = df_flare[['Jz', 'USFLUX', 'LOS','USJH','TOTPOT']].mean(axis = 1)
avg_in_noflare = df_noflare[['Jz', 'USFLUX', 'LOS','USJH','TOTPOT']].mean(axis = 1)

avg_in = pd.concat([avg_in_noflare, avg_in_flare], axis = 1)
avg_in = torch.nn.Softmax(dim = 1)(torch.tensor(avg_in.to_numpy()))
avg_in_lab = np.argmax(avg_in.numpy(), axis = 1)

cm = confusion_matrix(labels, avg_in_lab)
auc = roc_auc_score(labels, avg_in.numpy()[:,1])

acc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
prec = cm[1,1]/(cm[1,1]+cm[0,1])
rec = cm[1,1]/(cm[1,1]+cm[1,0])
f1 = (2* prec * rec)/(prec + rec)
tss = rec - cm[0,1]/(cm[0,0] + cm[0,1])

print('Confusion matrix: \n', cm)
print('AUC: ', auc)
print('Accuracy: ', acc)
print('Precision: ', prec)
print('Recall: ', rec)
print('F1-score: ', f1)
print('TSS: ', tss)
