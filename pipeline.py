import numpy as np
import pandas as pd
import pylab as plt
import os
import sys

import warnings
warnings.simplefilter('ignore')

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from scipy.stats import spearmanr

# define functions
def parity_plot(actual, predict, title, dirname = 'figures', compare = 'K'):
    fig, axes = plt.subplots(1, 1, figsize = (5, 4.5), dpi = 150)
    
    if compare == 'K':
        xlabel = 'GCMC simulated $K_H$ [mol/kg/Pa]'
        ylabel = 'ML predicted $K_H$ [mol/kg/Pa]'
        scale = 'log'

        space = np.array([1e-62, 1e12])
        ticks = np.logspace(-60, 10, 8)

        axes.plot([1e-15, 1e-15], space, '--k', linewidth = .5)
        axes.plot(space, [1e-15, 1e-15], '--k', linewidth = .5)

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

        axes.plot([1, 1], space, '--k', linewidth = .5)
        axes.plot(space, [1, 1], '--k', linewidth = .5)

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

def henry_at_diff_temp(k1, h, T1, T2):
    kB = 8.31446261815324
    k2 = k1 + h * 1000 / kB * (1 / T1 - 1 / T2)

    return k2

def r2_score_tilt(true, predict):
    true = pd.Series(true)
    predict = pd.Series(predict)
    true_index = pd.concat([true, predict], axis = 1).applymap(lambda x: x > -15).all(axis = 1)
    return r2_score(true.loc[true_index], predict.loc[true_index])

def mape(true, predict):
    true = pd.Series(true)
    predict = pd.Series(predict)
    #true_index = pd.concat([true, predict], axis = 1).applymap(lambda x: x > -15).all(axis = 1)
    #return np.mean(np.abs((true.loc[true_index] - predict.loc[true_index]) / true.loc[true_index]))
    return np.mean(np.abs((true - predict) / true))

def objective(true, predict):
    grad = np.tanh(predict - true)
    hess = 1 - np.tanh(predict - true)**2
    return grad, hess

def train_model(data, test_fold, compare = 'K'):
    print('Training on {} prediction\n'.format(compare))

    model = GradientBoostingRegressor(learning_rate = 0.1, loss = 'ls')

    param_grid = {'n_estimators': [1000],
    'max_depth': [3, 4, 5]}

    ps = PredefinedSplit(test_fold)

    if compare == 'K':
        scorer = make_scorer(mape, greater_is_better = False)
        gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, scoring = scorer, refit = True)
        gcv.fit(data[all_des].values, np.log10(data.K))
    
    elif compare == 'H':
        scorer = make_scorer(mean_absolute_error, greater_is_better = False)
        gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, scoring = scorer, refit = True)
        gcv.fit(data[all_des].values, data.H)

    best_model = gcv.best_estimator_
    
    print('Optimal hyperparameters: {}'.format(gcv.best_params_))

    return best_model
    #return gcv.best_params_

# make direcotries
figure = 'figures'
if not os.path.isdir(figure):
    os.makedirs(figure)

TSV = 'TSVs'
if not os.path.isdir(TSV):
    os.makedirs(TSV)

# read in the train/test set
train_test = pd.read_csv(sys.argv[1])

# read in the validation set
valid = pd.read_csv(sys.argv[2])
k_valid = np.log10(valid.K)
h_valid = valid.H

mofs = set(train_test.MOF)
molecules_train_test = set(train_test.molecule)
molecules_valid = set(valid.molecule)

# descriptors
text = train_test.columns[2:5]
pes = train_test.columns[5:33]
aprdf = train_test.columns[33:101]
mol = train_test.columns[101:137]
all_des = train_test.columns[2:137]

# ensemble modeling
random_seed = np.arange(1, 20, 2)
#random_seed = [1]

ensemble_K_test = []
ensemble_H_test = []

ensemble_K_valid = []
ensemble_H_valid = []

for i, seed in enumerate(random_seed):
    print('Split #{}\n'.format(i + 1))
    # make subdirecotries
    figure_dir = '{}/{}'.format(figure, i + 1)
    TSV_dir = '{}/{}'.format(TSV, i + 1)

    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    if not os.path.isdir(TSV_dir):
        os.makedirs(TSV_dir)

    train, test = train_test_split(train_test, test_size = .2, stratify = train_test.molecule, random_state = seed)

    k_train = np.log10(train.K)
    k_test = np.log10(test.K)

    h_train = train.H
    h_test = test.H

    tr_train, tr_test = train_test_split(train, test_size = .2, stratify = train.molecule, random_state = 42 - seed)

    test_fold = np.zeros(train.shape[0])
    for i in tr_train.index:
        a = train.index.get_loc(i)
        test_fold[a] = -1

    # predict the Henry's constants
    best_model_K = train_model(train, test_fold)

    k_train_predict = pd.Series(best_model_K.predict(train[all_des].values), index = k_train.index)
    k_test_predict = pd.Series(best_model_K.predict(test[all_des].values), index = k_test.index)

    print('r2 of K prediction for the training set: {}'.format(best_model_K.score(train[all_des].values, k_train)))
    print('r2 of K prediction for the test set: {}\n'.format(best_model_K.score(test[all_des].values, k_test)))

    print('MAPE of K prediction for the training set: {}'.format(mape(k_train, k_train_predict)))
    print('MAPE of K prediction for the test set: {}\n'.format(mape(k_test, k_test_predict)))

    parity_plot(k_train, k_train_predict, title = 'Training Set', dirname = figure_dir)
    parity_plot(k_test, k_test_predict, title = 'Test Set', dirname = figure_dir)

    k_train_csv = train[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_train_predict, columns = ['predict']))
    k_train_csv.to_csv('{}/{}.tsv'.format(TSV_dir, 'K_training'), sep = '\t', index = False)

    #k_test_csv = tr_test[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_tr_test_predict, columns = ['predict']))
    #k_test_csv.to_csv('{}/{}.tsv'.format(TSV_dir, 'K_test'), sep = '\t', index = False)

    # predict the heats of adsorption
    best_model_H = train_model(train, test_fold, compare = 'H')

    h_train_predict = pd.Series(best_model_H.predict(train[all_des].values), index = h_train.index)
    h_test_predict = pd.Series(best_model_H.predict(test[all_des].values), index = h_test.index)

    print('r2 of H prediction for the training set: {}'.format(best_model_H.score(train[all_des].values, h_train)))
    print('r2 of H prediction for the test set: {}\n'.format(best_model_H.score(test[all_des].values, h_test)))

    print('MAE of H prediction for the training set: {}'.format(mean_absolute_error(h_train, h_train_predict)))
    print('MAE of H prediction for the test set: {}\n'.format(mean_absolute_error(h_test, h_test_predict)))

    parity_plot(h_train, h_train_predict, title = 'Training Set', dirname = figure_dir, compare = 'H')
    parity_plot(h_test, h_test_predict, title = 'Test Set', dirname = figure_dir, compare = 'H')

    h_train_csv = train[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_train_predict, columns = ['predict']))
    h_train_csv.to_csv('{}/{}.tsv'.format(TSV_dir, 'H_training'), sep = '\t', index = False)

    #h_test_csv = tr_test[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_tr_test_predict, columns = ['predict']))
    #h_test_csv.to_csv('{}/{}.tsv'.format(TSV_dir, 'H_test'), sep = '\t', index = False)

    k_test_predict_values = k_test_predict.values
    h_test_predict_values = h_test_predict.values

    ensemble_K_test.append(k_test_predict_values)
    ensemble_H_test.append(h_test_predict_values)

    split_mol = open('{}/r2_test_molecule.tsv'.format(TSV_dir), 'w')
    split_mol.write('molecule\tr2\n')

    for mol in molecules_train_test:
        mol_test = k_test.loc[test.molecule == mol]
        mol_predict = k_test_predict.loc[test.molecule == mol]
        parity_plot(mol_test, mol_predict, title = '{}'.format(mol), dirname = '{}/test/molecule'.format(figure_dir))
        split_mol.write('{}\t{}\n'.format(mol, r2_score(mol_test, mol_predict)))

    split_mol.close()

    split_mof = open('{}/r2_test_mof.tsv'.format(TSV_dir), 'w')
    split_mof.write('MOF\tr2\n')

    for mof in mofs:
        mof_test = k_test.loc[test.MOF == mof]
        mof_predict = k_test_predict.loc[test.MOF == mof]

        if mof_test.shape[0] > 0:
        	parity_plot(mof_test, mof_predict, title = '{}'.format(mof), dirname = '{}/test/MOF'.format(figure_dir))
        	split_mof.write('{}\t{}\n'.format(mof, r2_score(mof_test, mof_predict)))

    split_mof.close()

    # test on the validation set
    k_valid_predict_values = best_model_K.predict(valid[all_des].values)
    h_valid_predict_values = best_model_H.predict(valid[all_des].values)

    ensemble_K_valid.append(k_valid_predict_values)
    ensemble_H_valid.append(h_valid_predict_values)

# average on the test set
'''
ensemble_K = np.array(ensemble_K_test)
k_test_predict_avg = np.mean(ensemble_K, axis = 0)
k_test_predict = test[['MOF', 'molecule', 'K']].join(pd.DataFrame(ensemble_K.T, columns = np.arange(1, len(random_seed) + 1, 1), index = k_test.index))
k_test_predict['average'] = k_test_predict_avg
k_test_predict['predict'] = 10.**k_test_predict_avg
print('r2 of K prediction for the test set: {}'.format(r2_score(k_test, k_test_predict_avg)))
print('MAPE of K prediction for the test set: {}\n'.format(mape(k_test, k_test_predict_avg)))
parity_plot(k_test, k_test_predict_avg, title = 'Test Set')
k_test_predict.to_csv('{}/{}.tsv'.format(TSV, 'K_test'), sep = '\t', index = False)

tsv_mol = open('TSVs/r2_test_molecule.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_train_test:
    mol_test = k_test.loc[test.molecule == mol]
    mol_average = k_test_predict['average'].loc[test.molecule == mol]
    parity_plot(mol_test, mol_average, title = '{}'.format(mol), dirname = 'figures/test/molecule', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score(mol_test, mol_average)))

tsv_mol.close()

tsv_mof = open('TSVs/r2_test_mof.tsv', 'w')
tsv_mof.write('MOF\tr2\n')

for mof in mofs:
    mof_test = k_test.loc[test.MOF == mof]
    mof_average = k_test_predict['average'].loc[test.MOF == mof]

    if mof_test.shape[0] > 0:
        parity_plot(mof_test, mof_average, title = '{}'.format(mof), dirname = 'figures/test/MOF', compare = 'K')
        tsv_mof.write('{}\t{}\n'.format(mof, r2_score(mof_test, mof_average)))

tsv_mof.close()

ensemble_H = np.array(ensemble_H_test)
h_test_predict_avg = np.mean(ensemble_H, axis = 0)
h_test_predict = test[['MOF', 'molecule', 'H']].join(pd.DataFrame(ensemble_H.T, columns = np.arange(1, len(random_seed) + 1, 1), index = h_test.index))
h_test_predict['average'] = h_test_predict_avg
print('r2 of H prediction for the test set: {}'.format(r2_score(h_test, h_test_predict_avg)))
print('MAE of H prediction for the test set: {}\n'.format(mean_absolute_error(h_test, h_test_predict_avg)))
parity_plot(h_test, h_test_predict_avg, title = 'Test Set', compare = 'H')
h_test_predict.to_csv('{}/{}.tsv'.format(TSV, 'H_test'), sep = '\t', index = False)
'''

# average on the validation set
ensemble_K = np.array(ensemble_K_valid)
k_valid_predict_avg = np.mean(ensemble_K, axis = 0)
k_valid_predict = valid[['MOF', 'molecule', 'K']].join(pd.DataFrame(ensemble_K.T, columns = np.arange(1, len(random_seed) + 1, 1), index = k_valid.index))
k_valid_predict['average'] = k_valid_predict_avg
k_valid_predict['predict'] = 10.**k_valid_predict_avg
print('r2 of K prediction for the validation set: {}'.format(r2_score(k_valid, k_valid_predict_avg)))
print('MAPE of K prediction for the validation set: {}\n'.format(mape(k_valid, k_valid_predict_avg)))
parity_plot(k_valid, k_valid_predict_avg, title = 'Validation Set')
k_valid_predict.to_csv('{}/{}.tsv'.format(TSV, 'K_validation'), sep = '\t', index = False)

tsv_mol = open('TSVs/r2_validation.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_valid:
    parity_plot(k_valid.loc[valid.molecule == mol], k_valid_predict['average'].loc[valid.molecule == mol], title = '{}'.format(mol), dirname = 'figures/validation/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score_tilt(k_valid.loc[valid.molecule == mol], k_valid_predict['average'].loc[valid.molecule == mol])))

tsv_mol.close()

ensemble_H = np.array(ensemble_H_valid)
h_valid_predict_avg = np.mean(ensemble_H, axis = 0)
h_valid_predict = valid[['MOF', 'molecule', 'H']].join(pd.DataFrame(ensemble_H.T, columns = np.arange(1, len(random_seed) + 1, 1), index = h_valid.index))
h_valid_predict['average'] = h_valid_predict_avg
print('r2 of H prediction for the validation set: {}'.format(r2_score(h_valid, h_valid_predict_avg)))
print('MAE of H prediction for the validation set: {}\n'.format(mean_absolute_error(h_valid, h_valid_predict_avg)))
parity_plot(h_valid, h_valid_predict_avg, title = 'Validation Set', compare = 'H')
h_valid_predict.to_csv('{}/{}.tsv'.format(TSV, 'H_validation'), sep = '\t', index = False)

'''
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
## training set
tsv_mol = open('TSVs/r2_train.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_train_test:
    parity_plot(k_train.loc[train.molecule == mol], k_train_predict.loc[train.molecule == mol], title = '{}'.format(mol), dirname = 'figures/train/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score_tilt(k_train.loc[train.molecule == mol], k_train_predict.loc[train.molecule == mol])))

tsv_mol.close()

## test set
tsv_mol = open('TSVs/r2_test.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_train_test:
    parity_plot(k_test.loc[test.molecule == mol], k_test_predict.loc[test.molecule == mol], title = '{}'.format(mol), dirname = 'figures/test/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score_tilt(k_test.loc[test.molecule == mol], k_test_predict.loc[test.molecule == mol])))

tsv_mol.close()

## validation set
tsv_mol = open('TSVs/r2_validation.tsv', 'w')
tsv_mol.write('molecule\tr2\n')

for mol in molecules_valid:
    parity_plot(k_valid.loc[valid.molecule == mol], k_valid_predict.loc[valid.molecule == mol], title = '{}'.format(mol), dirname = 'figures/validation/molecules', compare = 'K')
    tsv_mol.write('{}\t{}\n'.format(mol, r2_score_tilt(k_valid.loc[valid.molecule == mol], k_valid_predict.loc[valid.molecule == mol])))

tsv_mol.close()

# save feature importances
feat_import = best_model.feature_importances_
feat = pd.DataFrame(data = {'feature': all_des, 'feature importance': feat_import})
feat.to_csv('TSVs/feature_importance.csv', index = False)
'''
