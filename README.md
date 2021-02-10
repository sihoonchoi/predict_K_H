# predict_K_H

This repo provides a single pipeline with which Henry's constants and other properties of interest can be predicted for a given data of MOFs and adsorbates. `set_1.csv` serves as train/test set consisting of 471 MOFs and 30 molecules, while `set_2.csv` is a validation set with the same MOFs but new 15 molecules. `set_2.csv` is never used for the model training procedure.

Before running the code, modify line # 164 and # 165 where the regression model and its hyperparamter grid should be defined, respectively.
After setting a model and corresponding hyperparameter candidates, type the following:
`python pipeline.py set_1.csv set_2.csv`

Resulting files consist of:
- Parity plots of K regression for train/test/validation set
- Parity plots of selectivities prediction for near-azeotropic pairs in the validation set
- A CSV file of Spearman's coefficients of predicted selectivities for near-azeotropic pairs in the validation set
- Parity plots of H regression for train/test/validation set
- A CSV file of Spearman's coefficients of calculated selectivities at 373 K for near-azeotropic pairs in the validation set
