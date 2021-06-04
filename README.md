# predict_K_H

This repo provides codes and scripts in which generate descriptors and predict Henry's constants and the heats of adsorption for a given set of MOFs and adsorbates[1]. There are three directories: `descriptor_calculation`, `classification`, and `ensemble_modeling`.

## `descriptor_calculation`
This directory contains source codes of generating AP-RDF and molecular descriptors that are used in this study.

## `classification`
This directory serves as a source code for Section 3.I. `Data_S1.csv` is a train/test set consisting of 471 MOFs and 30 molecules, while `Data_S2.csv` is a hold-out set with the same MOFs but new 15 molecules. `Data_S2.csv` is never faced during the classification training procedure and is for assessing generalizability of our scheme throughout this work.

## `ensemble_modeling`
This directory is a source code for Section 3.II. `set_1.csv` and `set_2.csv` are the result of Section 3.I. from `Data_S1.csv` and `Data_S2.csv`, respectively. `set_1.csv` contains 12,960 MOF-molecule pairs and `set_2.csv` has 7,065 pairs.

To run the ensemble model, type the following command on Terminal: `python pipeline.py set_1.csv set_2.csv`

This will create one subdirectory: `data`

`data` directory consists of: 
- Predicted results for the training and test sets in each data split (numbered directories correspond to the random seeds used during the data split)
- Predicted results for set 2

[1] **X. Yu, S. Choi, D. Tang, A. J. Medford, D. Sholl, Efficient Models For Predicting Temperature-dependent Henry’s Constants and Adsorption Selectivities for Diverse Collections of Molecules in Metal-Organic Frameworks (Submitted)**

## Dependencies
- NumPy
- SciPy
- Pandas
- scikit-learn
