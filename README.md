# predict_K_H

This repo provides a single pipeline in which Henry's constants and the heats of adsorption can be predicted for a given data of MOFs and adsorbates. `set_1.csv` serves as a train/test set consisting of 471 MOFs and 30 molecules, while `set_2.csv` is a hold-out set with the same MOFs but new 15 molecules. `set_2.csv` is never used during the model training procedure and is for assessing generalizability of the model to unexperienced molecules.

To run the ML model, type the following command on Terminal: `python pipeline.py set_1.csv set_2.csv`

This will create one subdirectory: `data`

`data` directory consists of: 
- Predicted results for the training and test sets in each data split (numbered directories correspond to the random seeds used during the data split)
- Predicted results for set 2

In case of using our pipeline, please cite the article: **X. Yu, S. Choi, D. Tang, A. J. Medford, D. Sholl, Efficient Models For Predicting Temperature-dependent Henry’s Constants and Adsorption Selectivities for Diverse Collections of Molecules in Metal-Organic Frameworks (Submitted)**

## Dependencies
- NumPy
- SciPy
- Pandas
- scikit-learn
