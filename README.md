# SF-forecasting-FM 
Forecasting solar flares using the spatial distributions or field maps of different properties.  
This repository is accompanied by the publication:
> [Shenoy, A.N., H.N., A., Vijayasenan, D. et al. Solar Flare Forecasting Using the Spatial Distributions of Active Region Properties. Sol Phys 301, 161 (2026).](https://link.springer.com/article/10.1007/s11207-026-02751-w)
## Summary
|sav_nm|Model name|Inputs|
| --- | --- | --- |
| Vmgm | VectMgm | Bp, Br, Bt magnetograms |
| JFP | JzFluxPot | vertical current, unsigned magnetic flux, photospheric magnetic free energy density|
| FP | FluxPot | unsigned magnetic flux, photospheric magnetic free energy density|
| LOS | LOS | line-of-sight magnetic field |
| Jz | Jz | vertical current | 
| USJH | USJH | unsigned current helicity |
| USFLUX | USFLUX | unsigned magnetic flux | 
| TOTPOT | TOTPOT | photospheric magnetic free energy density |


## Data
- SHARP data
- Hollanda et al
- Data split

## Installation
Download the repository
```bash
git clone https://github.com/ANShenoy536/SF-forecasting-FM.git
```
Create a fresh conda environment using the yml file
```bash
conda env create -f CondaEnvironment.yml
```
## Pre-processing
This involes the generation of the spatial distributions or field maps of different properties of ARs using the SHARP vector magnetograms, and preparing them for our model. Follow the instructions in the README.md there.
## Training
We strart with the imagenet-pretrained model (with same or modeified layer one weights based on the input channels). We load the entrie model. We train the last layer with our data for a few epochs and then the entire network. We save and evaluate on the test set at regular intervals (controlled by epochs2). The performance metrics after each evaluation is saved in a csv file, and the corresponding predictions, loss curve, roc-auc curve are stored as well.

Before training, we need to create a csv file to save the results while training. Run the following script before training for a particular sav_num for the first time (ONLY THE 1st TIME for EACH sav_nm)
```bash
python training/make_res_file.py sav_nm sav_dir_pth
```
- `sav_nm` : string : is the token for the model as show in the table
- `sav_dir_path` : string : is the path where the generated csv will be stored

```bash
python training/SF_ResNet18_df_3ch.py <sav_nm> <set> <num> <epochs1> "<epochs2>" <learning_rate> <pre_trained_model_path>'
```
- `sav_nm` : string : is the token for the model as shown in the table
- `set` : int : is the five fold cross-validation set
- `num` : int : is the model number you want to assign
- `epochs1`: int: is number of epochs to train the last layer with all the other layers frozen
- `epochs2`: array : This array is provided within double quotes. The sum of all its entries is the number of epochs to train the entire network. Each entry indicates the number of epochs after which to save the model. Eg. "[5,5,2]" trains the entire network for a total of 12 epochs and saves the model and evaluates it after the 5th, 10th (5+5) and 12th (5+5+2) epoch.
- `learning_rate`: float : is the learning rate
- `pre_trained_model_path` : string : is the path to the pre-trained imagenet ResNet18 model.

For 2- and 1-channels run `SF_ResNet18_df_2ch.py` and `SF_ResNet18_df_1ch.py` respectively.
## Eval
After the five-fold cross validation, the best models for each fold, and the final performance metric can be obtained by running
```bash
python eval/evaluate_5foldCV_results.py <res_path>
```
- `res_path` : string : is the path to the results file generated after training
## Inference

## Links
If you are using this work please cite 

@article{shenoy2026solar,
  title={Solar Flare Forecasting Using the Spatial Distributions of Active Region Properties},
  author={Shenoy, Ashwini Nagaraj and HN, Adithya and Vijayasenan, Deepu and Padinhatteeri, Sreejith and Raghavendra, BS},
  journal={Solar Physics},
  volume={301},
  number={10},
  pages={161},
  year={2026},
  doi={https://doi.org/10.1007/s11207-026-02751-w}
  publisher={Springer}
}



#### Acknowledgements
1. Param utkarsh
2. Monica Bobra
