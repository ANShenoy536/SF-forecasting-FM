###############################
# evaluate_5foldCV_results.py
###############################
# This script finds the best models for each five-fold CV set, and then evaluates the average TSS and F-score for all the five folds.

import pandas as pd
import numpy as np
import sys

def get_best(df):
    df_best = pd.DataFrame()
    for Set in range(1,6):
        df_temp = df[df['Set'] == Set]
        df_best  = pd.concat([df_best, df_temp[df_temp['TSS ']  == np.max(df_temp['TSS '].values)]], axis = 0)
        avg_tss = np.mean(df_best['TSS '].values)
        avg_fscore = np.mean(df_best[' F-score'].values)
    return list(df_best['Model#'].values), avg_tss, avg_fscore


if len(sys.argv) == 2:
  res_path = argv[1]
else:
  print('Enter the command line input as: \n\t\tevaluate_5foldCV_results.py res_path')
  sys.exit(0)

df = pd.read_csv(res_path)
best_models, tss, fscore = get_best(df)
print('Average TSS: ', tss)
print('Average F-score: ', fscore)
print('Best models: ', best_models)
