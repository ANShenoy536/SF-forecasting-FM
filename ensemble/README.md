To generate the inputs for Average ensemble run the following, once for Flare and once for NoFlare. Make sure you update `all_models` in the script first.
```shell
python GenInput_5fold_Avg.py cls
```
- `cls`: string: 'Flare' or 'NoFlare'

Then run
```shell
python AvgEns_5fold.py
```

To generate the inputs for Ensemble with meta-learner, generate the data first as shown below. Make sure you update `all_models` in the script first.
```shell
python GenInput_5fold_Ens.py
python SF_Ensemble.py Set num epochs learning_rate
```
- `Set`: int : is the five fold cross-validation set
- `num`: int : is the model number you want to assign
- `epochs`: int : is number of epochs to train the model
- `learning_rate`: float : is the learning rate

<span style="font-size: 1.5em;">**Note:** The P(Flare) and P(NoFlare) values are actually, Logit(Flare) and Logit(NoFlare).</span>
