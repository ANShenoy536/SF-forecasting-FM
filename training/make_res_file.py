##################
# make_res_file.py
##################
# Creates a Results<sav_nm>.csv file with the required headers. This script needs to be run before the first time a model for a particular savnum is trained
import sys

if len(sys.argv) == 3:
  sav_nm = sys.argv[1]
  sav_dir_pth = sys.argv[2]
else:
  print('Enter the command line input as: \n\t\tmake_res_file.py sav_nm sav_dir_pth')

f = open(os.path.join(sav_dir_pth, 'Results' + sav_nm + '.csv'),'a')
f.write("Set,Model#,FM, LR, TP, TN, FP, FN,Accuracy, AUC, Precision, Recall, F-score,TSS \n")
f.close()
print('Done!')
