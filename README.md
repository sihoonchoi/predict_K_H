# predict_K_H

This repo provides a single pipeline in which Henry's constants and other properties of interest can be predicted for a given data of MOFs and adsorbates. `set_1.csv` serves as a train/test set consisting of 471 MOFs and 30 molecules, while `set_2.csv` is a validation set with the same MOFs but new 15 molecules. `set_2.csv` is never used during the model training procedure and is for assessing generalizability of the model to unexperienced molecules.

**Before running the code, modify line #99 - #100 where the regression model and its hyperparamter grid should be defined, respectively.**

After setting a model and corresponding hyperparameter candidates, type the following command on Terminal:  
`python pipeline.py set_1.csv set_2.csv`

This will create two subdirectories: `figures` and `CSVs`.

`figures` directory consists of:
- Parity plots of K regression for train/test/validation set
- Parity plots of selectivities prediction for 3 near-azeotropic pairs in the validation set
- Parity plots of H regression for train/test/validation set

`CSVs` directory consists of: 
- A CSV file of Spearman's coefficients of predicted selectivities for 3 near-azeotropic pairs in the validation set
- A CSV file of Spearman's coefficients of calculated selectivities at 373 K for 3 near-azeotropic pairs in the validation set
