import numpy as np
import pandas as pd
import pylab as plt
import os
import sys

import warnings
warnings.simplefilter('ignore')

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import r2_score, mean_absolute_error, max_error
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from itertools import combinations

# define functions
def parity_plot(actual, predict, title, dirname = 'figures', compare = 'K'):
    fig, axes = plt.subplots(1, 1, figsize = (5, 4.5), dpi = 150)
    
    if compare == 'K':
        xlabel = 'Actual $K_H$ [mol/kg/Pa]'
        ylabel = 'Predicted $K_H$ [mol/kg/Pa]'
        scale = 'log'

        space = np.array([1e-62, 1e12])
        ticks = np.logspace(-60, 10, 8)

        axes.scatter(10.**actual, 10.**predict, marker =  '.', alpha = .4)
        
    elif compare == 'H':
        xlabel = 'Actual $\Delta H_{ads}$ [kJ/mol]'
        ylabel = 'Predicted $\Delta H_{ads}$ [kJ/mol]'
        scale = 'linear'

        space = np.array([-25, 150])
        ticks = np.linspace(0, 150, 4)

        axes.scatter(actual, predict, marker =  '.', alpha = .4)

    elif compare == 'selectivity':
        xlabel = 'Actual Selectivity'
        ylabel = 'Predicted Selectivity'
        scale = 'log'

        space_min = np.floor(min(actual.min(), predict.min()))
        space_max = np.ceil(max(actual.max(), predict.max()))
        space = np.array([10.**space_min, 10.**space_max])
        ticks = np.logspace(-60, 10, 8)

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

def train_model(model, param_grid, X_train, X_test, y_train, y_test, compare = 'K'):
    print('Training on {} prediction\n'.format(compare))

    X = np.append(X_train, X_test, axis = 0)
    y = y_train.append(y_test)

    test_fold = np.zeros(X.shape[0])
    test_fold[:X_train.shape[0]] = -1
    ps = PredefinedSplit(test_fold)
    
    gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, verbose = 1)

    gcv.fit(X, y)
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
CSV = 'CSVs'
if not os.path.isdir(CSV):
    os.makedirs(CSV)

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
aprdf = train_test.columns[33:69]
mol = train_test.columns[69:137]
all_des = train_test.columns[2:137]

# properties of interest
k_train = np.log10(train.K)
k_test = np.log10(test.K)
k_valid = np.log10(valid.K)

h_train = train.H
h_test = test.H
h_valid = valid.H

# predict the Henry's constants
model = KernelRidge(kernel = 'rbf') # change this part
param_grid = {'alpha': np.logspace(-1, 1, 5), 'gamma': np.logspace(-2, 0, 5)}
best_model = train_model(model, param_grid, train[all_des], test[all_des], k_train, k_test)

k_train_predict = pd.Series(best_model.predict(train[all_des]), index = k_train.index)
k_test_predict = pd.Series(best_model.predict(test[all_des]), index = k_test.index)

parity_plot(k_train, k_train_predict, title = 'Training Set')
parity_plot(k_test, k_test_predict, title = 'Test Set')

# test on the validation set
k_valid_predict = pd.Series(best_model.predict(valid[all_des]), index = k_valid.index)
print('r2 of K prediction for the validation set: {}\n'.format(best_model.score(valid[all_des], k_valid)))
parity_plot(k_valid, k_valid_predict, title = 'Validation Set')

# analyze the near-azeotropic pairs
azeo_pairs = [('1-methyl-3-buten-1-ol', '3-methyl-1-butanol'), ('2-hexene', '1,5-hexadiene'), ('propionaldehyde', 'propylamine')]

csv = open('CSVs/spearman.csv', 'w')
csv.write('mol1,mol2,spearman\n')

for pair in azeo_pairs:
    mol1, mol2 = pair
    selectivity = get_selectivity(mol1, mol2, valid, k_valid, k_valid_predict)
    parity_plot(selectivity.simulation, selectivity.ML, title = '{} - {}'.format(mol1, mol2), compare = 'selectivity')
    csv.write('{},{},{}\n'.format(mol1, mol2, spearmanr(selectivity.simulation, selectivity.ML)[0]))

csv.close()

# predict the heats of adsorption
best_model_H = train_model(model, param_grid, train[all_des], test[all_des], h_train, h_test, compare = 'H')

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
print('Test Set\n')
print('MAE: {}'.format(mean_absolute_error(h_test, h_test_predict)))
print('Max: {}\n'.format(max_error(h_test, h_test_predict)))
print('Validation Set\n')
print('MAE: {}'.format(mean_absolute_error(h_valid, h_valid_predict)))
print('Max: {}\n'.format(max_error(h_valid, h_valid_predict)))

# calculate the selectivity at 373K
k_valid_373 = henry_at_diff_temp(k_valid, h_valid, 300, 373)
k_valid_predict_373 = henry_at_diff_temp(k_valid_predict, h_valid_predict, 300, 373)

csv_373 = open('CSVs/spearman_373.csv', 'w')
csv_373.write('mol1,mol2,spearman\n')

for pair in azeo_pairs:
    mol1, mol2 = pair
    selectivity_373 = get_selectivity(mol1, mol2, valid, k_valid_373, k_valid_predict_373)
    parity_plot(selectivity_373.simulation, selectivity_373.ML, title = '{} - {} at 373K'.format(mol1, mol2), compare = 'selectivity')
    csv_373.write('{},{},{}\n'.format(mol1, mol2, spearmanr(selectivity_373.simulation, selectivity_373.ML)[0]))

csv_373.close()