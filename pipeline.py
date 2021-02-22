import numpy as np
import pandas as pd
import pylab as plt
import os
import sys

import warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, max_error
from sklearn.utils import shuffle
from scipy.stats import spearmanr

# define functions
def parity_plot(actual, predict, title, dirname = 'figures', compare = 'K'):
    fig, axes = plt.subplots(1, 1, figsize = (5, 4.5), dpi = 150)
    
    if compare == 'K':
        xlabel = 'GCMC simulated $K_H$ [mol/kg/Pa]'
        ylabel = 'ML predicted $K_H$ [mol/kg/Pa]'
        scale = 'log'

        space = np.array([1e-22, 1e12])
        ticks = np.logspace(-20, 10, 7)

        axes.scatter(10.**actual, 10.**predict, marker =  '.', alpha = .4)
        
    elif compare == 'H':
        xlabel = 'GCMC simulated $\Delta H_{ads}$ [kJ/mol]'
        ylabel = 'ML predicted $\Delta H_{ads}$ [kJ/mol]'
        scale = 'linear'

        space = np.array([-50, 150])
        ticks = np.linspace(-50, 150, 5)

        axes.scatter(actual, predict, marker =  '.', alpha = .4)

    elif compare == 'selectivity':
        xlabel = 'GCMC simulated selectivity'
        ylabel = 'ML predicted selectivity'
        scale = 'log'

        space = np.array([1e-10, 1e10])
        ticks = np.logspace(-10, 10, 5)

        axes.scatter(10.**actual, 10.**predict, marker = '.', alpha = .4)

    axes.set_xscale(scale)
    axes.set_yscale(scale)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    axes.plot(space, space, 'k-')
    axes.set_title(title)
    axes.set_xlim(space)
    axes.set_ylim(space)
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)

    plt.tight_layout()

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    plt.savefig('{}/{}_{}.png'.format(dirname, title, compare))
    plt.close()

def get_selectivity(mol1, mol2, X, actual, predict):
    mol1_mof = set(X[X.molecule == mol1].MOF)
    mol2_mof = set(X[X.molecule == mol2].MOF)

    common_mof = list(mol1_mof & mol2_mof)

    if len(common_mof):
        mol1_sim = []
        mol2_sim = []
        mol1_ML = []
        mol2_ML = []

        for mof in common_mof:
            mol1_sim.append(actual[np.logical_and(X.molecule == mol1, X.MOF == mof)].values[0])
            mol2_sim.append(actual[np.logical_and(X.molecule == mol2, X.MOF == mof)].values[0])
            mol1_ML.append(predict[np.logical_and(X.molecule == mol1, X.MOF == mof)].values[0])
            mol2_ML.append(predict[np.logical_and(X.molecule == mol2, X.MOF == mof)].values[0])

        selectivity_sim = (np.array(mol1_sim) - np.array(mol2_sim)).reshape(-1,)
        selectivity_ML = (np.array(mol1_ML) - np.array(mol2_ML)).reshape(-1,)

        selectivity = pd.DataFrame({'MOF': common_mof, 'simulation': selectivity_sim, 'ML': selectivity_ML})

        return selectivity

def train_model(X_train, X_test, y_train, y_test, compare = 'K'):
    print('Training on {} prediction\n'.format(compare))
    
    model = GradientBoostingRegressor(max_depth = 3, learning_rate = 0.1)
    param_grid = {'n_estimators': np.array([5000]), 'loss': ['ls']}

    X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train)
    gcv = GridSearchCV(model, param_grid, cv = 3, n_jobs = -1, verbose = 1)
    gcv.fit(X_train_shuffle, y_train_shuffle)
    best_model = gcv.best_estimator_
        
    print('Optimal hyperparameters: {}'.format(gcv.best_params_))
    print('r2 of {} prediction for the training set: {}'.format(compare, best_model.score(X_train, y_train)))
    print('r2 of {} prediction for the test set: {}'.format(compare, best_model.score(X_test, y_test)))

    return best_model

def henry_at_diff_temp(k1, h, T1, T2):
    kB = 8.31446261815324
    k2 = k1 + h * 1000 / kB * (1 / T1 - 1 / T2)

    return k2

# make a directory to save figures
figure = 'figures'
if not os.path.isdir(figure):
    os.makedirs(figure)

# make a directory to save CSVs
TSV = 'TSVs'
if not os.path.isdir(TSV):
    os.makedirs(TSV)

# data organizing
train_test = pd.read_csv(sys.argv[1])
valid = pd.read_csv(sys.argv[2])

train_index = train_test.Set == 'train'
test_index = train_test.Set == 'test'

train = train_test.loc[train_index]
test = train_test.loc[test_index]

mofs = set(train_test.MOF)
molecules_train_test = set(train_test.molecule)
molecules_valid = set(valid.molecule)

# descriptors
text = train_test.columns[2:5]
pes = train_test.columns[5:33]
aprdf = train_test.columns[33:101]
mol = train_test.columns[101:137]
all_des = train_test.columns[2:137]

# properties of interest
k_train = np.log10(train.K)
k_test = np.log10(test.K)
k_valid = np.log10(valid.K)

h_train = train.H
h_test = test.H
h_valid = valid.H

# predict the Henry's constants

best_model = train_model(train[all_des], test[all_des], k_train, k_test)

k_train_predict = pd.Series(best_model.predict(train[all_des]), index = k_train.index)
k_test_predict = pd.Series(best_model.predict(test[all_des]), index = k_test.index)

parity_plot(k_train, k_train_predict, title = 'Training Set')
parity_plot(k_test, k_test_predict, title = 'Test Set')

# test on the validation set
k_valid_predict = pd.Series(best_model.predict(valid[all_des]), index = k_valid.index)
print('r2 of K prediction for the validation set: {}\n'.format(best_model.score(valid[all_des], k_valid)))
parity_plot(k_valid, k_valid_predict, title = 'Validation Set')

# predict the heats of adsorption
best_model_H = train_model(train[all_des], test[all_des], h_train, h_test, compare = 'H')

h_train_predict = pd.Series(best_model_H.predict(train[all_des]), index = h_train.index)
h_test_predict = pd.Series(best_model_H.predict(test[all_des]), index = h_test.index)

parity_plot(h_train, h_train_predict, title = 'Training Set', compare = 'H')
parity_plot(h_test, h_test_predict, title = 'Test Set', compare = 'H')

# test on the validation set
h_valid_predict = pd.Series(best_model_H.predict(valid[all_des]), index = h_valid.index)
print('r2 of H prediction for the validation set: {}\n'.format(best_model_H.score(valid[all_des], h_valid)))
parity_plot(h_valid, h_valid_predict, title = 'Validation Set', compare = 'H')

print('Training Set')
print('MAE: {}'.format(mean_absolute_error(h_train, h_train_predict)))
print('Max: {}\n'.format(max_error(h_train, h_train_predict)))
print('Test Set')
print('MAE: {}'.format(mean_absolute_error(h_test, h_test_predict)))
print('Max: {}\n'.format(max_error(h_test, h_test_predict)))
print('Validation Set')
print('MAE: {}'.format(mean_absolute_error(h_valid, h_valid_predict)))
print('Max: {}\n'.format(max_error(h_valid, h_valid_predict)))

# analyze the near-azeotropic pairs
## training set
azeo_pairs_train_test = [
    ('propene', 'propane'),
    ('acetonitrile', 'isopropyl alcohol'),
    ('propyl alcohol', 'methyl propyl ketone'),
    ('dimethylamine', 'acetaldehyde'),
    ('acetaldehyde', 'ethylamine'),
    ('1,5-heptadiene', 'propionitrile'),
    ('dimethylamine', 'ethylamine'),
    ('methyl propyl ether', '2-pentene'),
    ('neopentane', 'methyl isopropyl ether'),
    ('4-methyl-1-hexene', '4,4-dimethyl-1-pentene')
    ]

tsv = open('TSVs/spearman_train.tsv', 'w')
tsv.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_train_test:
    mol1, mol2 = pair
    selectivity = get_selectivity(mol1, mol2, train, k_train, k_train_predict)
    parity_plot(selectivity.simulation, selectivity.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/train/300', compare = 'selectivity')
    tsv.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity.simulation, selectivity.ML)[0]))

tsv.close()

## test set
tsv = open('TSVs/spearman_test.tsv', 'w')
tsv.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_train_test:
    mol1, mol2 = pair
    selectivity = get_selectivity(mol1, mol2, test, k_test, k_test_predict)
    parity_plot(selectivity.simulation, selectivity.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/test/300', compare = 'selectivity')
    tsv.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity.simulation, selectivity.ML)[0]))

tsv.close()

## validation set
azeo_pairs_valid = [
    ('1-methyl-3-buten-1-ol', '3-methyl-1-butanol'),
    ('2-hexene', '1,5-hexadiene'),
    ('propionaldehyde', 'propylamine')
    ]

tsv = open('TSVs/spearman_valid.tsv', 'w')
tsv.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_valid:
    mol1, mol2 = pair
    selectivity = get_selectivity(mol1, mol2, valid, k_valid, k_valid_predict)
    parity_plot(selectivity.simulation, selectivity.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/validation/300', compare = 'selectivity')
    tsv.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity.simulation, selectivity.ML)[0]))

tsv.close()

# calculate the selectivity at 373K
## training set
k_train_373 = henry_at_diff_temp(k_train, h_train, 300, 373)
k_train_predict_373 = henry_at_diff_temp(k_train_predict, h_train_predict, 300, 373)

tsv_373 = open('TSVs/spearman_373_train.tsv', 'w')
tsv_373.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_train_test:
    mol1, mol2 = pair
    selectivity_373 = get_selectivity(mol1, mol2, train, k_train_373, k_train_predict_373)
    parity_plot(selectivity_373.simulation, selectivity_373.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/train/373', compare = 'selectivity')
    tsv_373.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity_373.simulation, selectivity_373.ML)[0]))

tsv_373.close()

## test set
k_test_373 = henry_at_diff_temp(k_test, h_test, 300, 373)
k_test_predict_373 = henry_at_diff_temp(k_test_predict, h_test_predict, 300, 373)

tsv_373 = open('TSVs/spearman_373_test.tsv', 'w')
tsv_373.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_train_test:
    mol1, mol2 = pair
    selectivity_373 = get_selectivity(mol1, mol2, test, k_test_373, k_test_predict_373)
    parity_plot(selectivity_373.simulation, selectivity_373.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/test/373', compare = 'selectivity')
    tsv_373.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity_373.simulation, selectivity_373.ML)[0]))

tsv_373.close()

## validation set
k_valid_373 = henry_at_diff_temp(k_valid, h_valid, 300, 373)
k_valid_predict_373 = henry_at_diff_temp(k_valid_predict, h_valid_predict, 300, 373)

tsv_373 = open('TSVs/spearman_373_valid.tsv', 'w')
tsv_373.write('mol1\tmol2\tspearman\n')

for pair in azeo_pairs_valid:
    mol1, mol2 = pair
    selectivity_373 = get_selectivity(mol1, mol2, valid, k_valid_373, k_valid_predict_373)
    parity_plot(selectivity_373.simulation, selectivity_373.ML, title = '{} - {}'.format(mol1, mol2), dirname = 'figures/validation/373', compare = 'selectivity')
    tsv_373.write('{}\t{}\t{}\n'.format(mol1, mol2, spearmanr(selectivity_373.simulation, selectivity_373.ML)[0]))

tsv_373.close()

# r2 on molecule
## test set
tsv_mol = open('TSVs/r2_test.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_train_test:
    parity_plot(k_test.loc[test.molecule == mol], k_test_predict.loc[test.molecule == mol], title = '{}'.format(mol), dirname = 'figures/test/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score(k_test.loc[test.molecule == mol], k_test_predict.loc[test.molecule == mol])))

tsv_mol.close()

## validation set
tsv_mol = open('TSVs/r2_validation.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_valid:
    parity_plot(k_valid.loc[valid.molecule == mol], k_valid_predict.loc[valid.molecule == mol], title = '{}'.format(mol), dirname = 'figures/validation/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score(k_valid.loc[valid.molecule == mol], k_valid_predict.loc[valid.molecule == mol])))

tsv_mol.close()

# save predicted values
## Henry's constants
k_train_together = train[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_train_predict, columns = ['K_predict']))
k_train_together.to_csv('TSVs/{}.tsv'.format('K_training'), sep = '\t', index = False)

k_test_together = test[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_test_predict, columns = ['K_predict']))
k_test_together.to_csv('TSVs/{}.tsv'.format('K_test'), sep = '\t', index = False)

k_valid_together = valid[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_valid_predict, columns = ['K_predict']))
k_valid_together.to_csv('TSVs/{}.tsv'.format('K_validation'), sep = '\t', index = False)

## heats of adsorption
h_train_together = train[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_train_predict, columns = ['H_predict']))
h_train_together.to_csv('TSVs/{}.tsv'.format('H_training'), sep = '\t', index = False)

h_test_together = test[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_test_predict, columns = ['H_predict']))
h_test_together.to_csv('TSVs/{}.tsv'.format('H_test'), sep = '\t', index = False)

h_valid_together = valid[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_valid_predict, columns = ['H_valid']))
h_valid_together.to_csv('TSVs/{}.tsv'.format('H_valid'), sep = '\t', index = False)
