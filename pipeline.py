import numpy as np
import pandas as pd
import pylab as plt
import os
import sys

import warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from scipy.stats import spearmanr

# plot settings
plt.rcParams['mathtext.default'] = 'regular'

# define functions
def train_model(data, test_fold, compare):
    print('Training on {} prediction\n'.format(compare))

    model = GradientBoostingRegressor(learning_rate = 0.1, loss = 'ls')

    param_grid = {'n_estimators': [4000],
    'max_depth': [5]}

    ps = PredefinedSplit(test_fold)
    scorer = make_scorer(mean_absolute_error, greater_is_better = False)

    if compare == 'K':
        gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, scoring = scorer, refit = True)
        gcv.fit(data[all_des].values, np.log10(data.K))
    
    elif compare == 'H':
        gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, scoring = scorer, refit = True)
        gcv.fit(data[all_des].values, data.H)

    best_model = gcv.best_estimator_
    
    print('Optimal hyperparameters: {}\n'.format(gcv.best_params_))

    return best_model

def r2_score_tilt(true, predict):
    true = pd.Series(true)
    predict = pd.Series(predict)
    true_index = pd.concat([true, predict], axis = 1).applymap(lambda x: x > -15).all(axis = 1)
    if true.loc[true_index].shape[0] > 0:
        return r2_score(true.loc[true_index], predict.loc[true_index])
    else:
        return float('nan')

def parity_plot(actual, predict, title, compare, dirname = 'figures'):
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

def henry_at_diff_temp(k1, h, T1, T2):
    kB = 8.31446261815324
    k2 = k1 + h * 1000 / kB * (1 / T1 - 1 / T2)

    return k2

# make direcotries
figure = 'figures'
if not os.path.isdir(figure):
    os.makedirs(figure)

TSV = 'TSVs'
if not os.path.isdir(TSV):
    os.makedirs(TSV)

# read in the train/validation set
train_valid = pd.read_csv(sys.argv[1])

# read in the test set
test = pd.read_csv(sys.argv[2])
k_test = np.log10(test.K)
h_test = test.H

mofs = set(train_valid.MOF)
molecules_train_valid = set(train_valid.molecule)
molecules_test = set(test.molecule)

# descriptors
text = train_valid.columns[2:5]
pes = train_valid.columns[5:33]
aprdf = train_valid.columns[33:101]
mol = train_valid.columns[101:137]
all_des = train_valid.columns[2:137]

# ensemble modeling
seed = 11
random_seed = np.arange(1, seed, 1)

ensemble_K_valid = []
ensemble_H_valid = []

ensemble_K_test = []
ensemble_H_test = []

for i, seed in enumerate(random_seed):
    print('split #{}\n'.format(i + 1))

    # make subdirecotries
    figure_split = '{}/{}'.format(figure, i + 1)
    TSV_split = '{}/{}'.format(TSV, i + 1)

    if not os.path.isdir(figure_split):
        os.makedirs(figure_split)
    if not os.path.isdir(TSV_split):
        os.makedirs(TSV_split)

    # split dataset
    train, valid = train_test_split(train_valid, test_size = .2, stratify = train_valid.molecule, random_state = seed)

    k_train = np.log10(train.K)
    k_valid = np.log10(valid.K)

    h_train = train.H
    h_valid = valid.H

    tr_train, tr_test = train_test_split(train, test_size = .2, stratify = train.molecule, random_state = seed)

    test_fold = np.zeros(train.shape[0])
    for i in tr_train.index:
        a = train.index.get_loc(i)
        test_fold[a] = -1

    # predict K
    best_model_K = train_model(train, test_fold, compare = 'K')

    k_train_predict = pd.Series(best_model_K.predict(train[all_des].values), index = k_train.index)
    k_valid_predict = pd.Series(best_model_K.predict(valid[all_des].values), index = k_valid.index)

    print('r2 of log(K) prediction for the training set: {}'.format(r2_score_tilt(k_train, k_train_predict)))
    print('r2 of log(K) prediction for the validation set: {}\n'.format(r2_score_tilt(k_valid, k_valid_predict)))

    print('MAE of log(K) prediction for the training set: {}'.format(mean_absolute_error(k_train, k_train_predict)))
    print('MAE of log(K) prediction for the validation set: {}\n'.format(mean_absolute_error(k_valid, k_valid_predict)))

    parity_plot(k_train, k_train_predict, title = 'train', compare = 'K', dirname = figure_split)
    parity_plot(k_valid, k_valid_predict, title = 'valid', compare = 'K', dirname = figure_split)

    k_train_csv = train[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_train_predict, columns = ['predict']))
    k_train_csv.to_csv('{}/{}.tsv'.format(TSV_split, 'train_K'), sep = '\t', index = False)

    k_valid_csv = valid[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_valid_predict, columns = ['predict']))
    k_valid_csv.to_csv('{}/{}.tsv'.format(TSV_split, 'valid_K'), sep = '\t', index = False)

    # predict H
    best_model_H = train_model(train, test_fold, compare = 'H')

    h_train_predict = pd.Series(best_model_H.predict(train[all_des].values), index = h_train.index)
    h_valid_predict = pd.Series(best_model_H.predict(valid[all_des].values), index = h_valid.index)

    print('r2 of H prediction for the training set: {}'.format(r2_score(h_train, h_train_predict)))
    print('r2 of H prediction for the validation set: {}\n'.format(r2_score(h_valid, h_valid_predict)))

    print('MAE of H prediction for the training set: {}'.format(mean_absolute_error(h_train, h_train_predict)))
    print('MAE of H prediction for the validation set: {}\n'.format(mean_absolute_error(h_valid, h_valid_predict)))

    parity_plot(h_train, h_train_predict, title = 'train', compare = 'H', dirname = figure_split)
    parity_plot(h_valid, h_valid_predict, title = 'valid', compare = 'H', dirname = figure_split)

    h_train_csv = train[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_train_predict, columns = ['predict']))
    h_train_csv.to_csv('{}/{}.tsv'.format(TSV_split, 'train_H'), sep = '\t', index = False)

    h_valid_csv = valid[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_valid_predict, columns = ['predict']))
    h_valid_csv.to_csv('{}/{}.tsv'.format(TSV_split, 'valid_H'), sep = '\t', index = False)

    # gather metrics by molecules
    split_mol = open('{}/metrics_valid_molecule.tsv'.format(TSV_split), 'w')
    split_mol.write('molecule\tr2\tspearman\tMAE\n')

    for mol in molecules_train_valid:
        mol_valid = k_valid.loc[valid.molecule == mol]
        mol_predict = k_valid_predict.loc[valid.molecule == mol]
        parity_plot(mol_valid, mol_predict, title = '{}'.format(mol), compare = 'K', dirname = '{}/valid/molecule'.format(figure_split))
        split_mol.write('{}\t{}\t{}\t{}\n'.format(mol, r2_score_tilt(mol_valid, mol_predict), spearmanr(mol_valid, mol_predict)[0], mean_absolute_error(mol_valid, mol_predict)))

    split_mol.close()

    # gather metrics by MOFs
    split_mof = open('{}/metrics_valid_mof.tsv'.format(TSV_split), 'w')
    split_mof.write('MOF\tr2\tspearman\tMAE\n')

    for mof in mofs:
        mof_valid = k_valid.loc[valid.MOF == mof]
        mof_predict = k_valid_predict.loc[valid.MOF == mof]

        if mof_valid.shape[0] > 0:
            parity_plot(mof_valid, mof_predict, title = '{}'.format(mof), compare = 'K', dirname = '{}/valid/MOF'.format(figure_split))
            split_mof.write('{}\t{}\t{}\t{}\n'.format(mof, r2_score_tilt(mof_valid, mof_predict), spearmanr(mof_valid, mof_predict)[0], mean_absolute_error(mof_valid, mof_predict)))

    split_mof.close()

    # gather predicted values on the validation set
    k_valid_predict_values = k_valid_predict.values
    h_valid_predict_values = h_valid_predict.values

    ensemble_K_valid.append(k_valid_predict_values)
    ensemble_H_valid.append(h_valid_predict_values)

    # gather predicted values on the test set
    k_test_predict_values = best_model_K.predict(test[all_des].values)
    h_test_predict_values = best_model_H.predict(test[all_des].values)

    ensemble_K_test.append(k_test_predict_values)
    ensemble_H_test.append(h_test_predict_values)

# average on the test set
## K prediction
ensemble_K = np.array(ensemble_K_test)
k_test_predict_avg = np.mean(ensemble_K, axis = 0)
k_test_predict = test[['MOF', 'molecule', 'K']].join(pd.DataFrame(ensemble_K.T, columns = np.arange(1, len(random_seed) + 1, 1), index = k_test.index))
k_test_predict['average'] = k_test_predict_avg
k_test_predict['predict'] = 10.**k_test_predict_avg
print('r2 of log(K) prediction for the test set: {}'.format(r2_score_tilt(k_test, k_test_predict_avg)))
print('MAE of log(K) prediction for the test set: {}\n'.format(mean_absolute_error(k_test, k_test_predict_avg)))
parity_plot(k_test, k_test_predict_avg, title = 'test', compare = 'K')
k_test_predict.to_csv('{}/{}.tsv'.format(TSV, 'test_K'), sep = '\t', index = False)

tsv_mol = open('TSVs/metrics_test_mol.tsv', 'w')
tsv_mol.write('molecule\tr2\tspearman\tMAE\n')

for mol in molecules_test:
    parity_plot(k_test.loc[test.molecule == mol], k_test_predict['average'].loc[test.molecule == mol], title = '{}'.format(mol), dirname = 'figures/test/molecule', compare = 'K')
    r2 = r2_score_tilt(k_test.loc[test.molecule == mol], k_test_predict['average'].loc[test.molecule == mol])
    S = spearmanr(k_test.loc[test.molecule == mol], k_test_predict['average'].loc[test.molecule == mol])[0]
    MAE = mean_absolute_error(k_test.loc[test.molecule == mol], k_test_predict['average'].loc[test.molecule == mol])
    tsv_mol.write('{}\t{}\t{}\t{}\n'.format(mol, r2, S, MAE))

tsv_mol.close()

## H prediction
ensemble_H = np.array(ensemble_H_test)
h_test_predict_avg = np.mean(ensemble_H, axis = 0)
h_test_predict = test[['MOF', 'molecule', 'H']].join(pd.DataFrame(ensemble_H.T, columns = np.arange(1, len(random_seed) + 1, 1), index = h_test.index))
h_test_predict['average'] = h_test_predict_avg
print('r2 of H prediction for the test set: {}'.format(r2_score(h_test, h_test_predict_avg)))
print('MAE of H prediction for the test set: {}\n'.format(mean_absolute_error(h_test, h_test_predict_avg)))
parity_plot(h_test, h_test_predict_avg, title = 'test', compare = 'H')
h_test_predict.to_csv('{}/{}.tsv'.format(TSV, 'test_H'), sep = '\t', index = False)

# near-azeotropic pairs
## training set
azeo_pairs_train_valid = [
    ('propionitrile', '1,5-heptadiene'),
    ('methyl isopropyl ether', 'neopentane'),
    ('propyl alcohol', 'methyl propyl ketone'),
    ('acetonitrile', 'isopropyl alcohol'),
    ('4-methyl-1-hexene', '4,4-dimethyl-1-pentene'),
    ('acetaldehyde', 'ethylamine'),
    ('acetaldehyde', 'dimethylamine'),
    ('methyl propyl ether', '2-pentene'),
    ('propene', 'propane'),
    ('ethylamine', 'dimethylamine')
    ]

## test set
azeo_pairs_test = [
    ('1-methyl-3-buten-1-ol', '3-methyl-1-butanol'),
    ('2-hexene', '1,5-hexadiene'),
    ('propionaldehyde', 'propylamine')
    ]

# Spearman's coefficients at 300 K
sel_valid_300 = open('{}/selectivity_valid_300.tsv'.format(TSV), 'w')
sel_valid_300.write('mol_1\tmol_2\tmean\tstd\n')

for j, pair in enumerate(azeo_pairs_train_valid):
    mol_1, mol_2 = pair

    selectivity = []
    for i in range(1, seed):
        df = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')

        mol_1_index = df[df.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
        df_1 = df[df.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]

        mol_2_index = df[df.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
        df_2 = df[df.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]

        join = df_1.join(df_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

        join['S_ML'] = join.predict_1 / join.predict_2
        join['S_GCMC'] = join.K_1 / join.K_2

        S = spearmanr(join.S_GCMC, join.S_ML)[0]
        selectivity.append(S)

    selectivity = np.array(selectivity)
    sel_valid_300.write('{}\t{}\t{}\t{}\n'.format(mol_1, mol_2, selectivity.mean(), selectivity.std()))

sel_valid_300.close()

# Spearman's coefficients at 373 K
sel_valid_373 = open('{}/selectivity_valid_373.tsv'.format(TSV), 'w')
sel_valid_373.write('mol_1\tmol_2\tmean\tstd\n')

for j, pair in enumerate(azeo_pairs_train_valid):
    mol_1, mol_2 = pair

    selectivity = []
    for i in range(1, seed):
        K = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
        H = pd.read_csv('{}/{}/valid_H.tsv'.format(TSV, i), sep = '\t')

        mol_1_index = K[K.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
        K_1 = K[K.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]
        H_1 = H[H.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['H', 'predict']]

        mol_2_index = K[K.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
        K_2 = K[K.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
        H_2 = H[H.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['H', 'predict']]

        K_1['K_373'] = henry_at_diff_temp(K_1.K, H_1.H, 300, 373)
        K_1['predict_373'] = henry_at_diff_temp(K_1.predict, H_1.predict, 300, 373)
        
        K_2['K_373'] = henry_at_diff_temp(K_2.K, H_2.H, 300, 373)
        K_2['predict_373'] = henry_at_diff_temp(K_2.predict, H_2.predict, 300, 373)

        join = K_1.join(K_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

        join['S_ML'] = join.predict_373_1 / join.predict_373_2
        join['S_GCMC'] = join.K_373_1 / join.K_373_2

        S = spearmanr(join.S_GCMC, join.S_ML)[0]
        selectivity.append(S)

    selectivity = np.array(selectivity)
    sel_valid_373.write('{}\t{}\t{}\t{}\n'.format(mol_1, mol_2, selectivity.mean(), selectivity.std()))

sel_valid_373.close()

# get figures
def figure3():
    fig, axes = plt.subplots(1, 4, figsize = (16, 4), dpi = 300)
    axes = axes.ravel()

    for i in range(1, 5):
        te = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
        GCMC = te.K
        ML = te.predict
        
        axes[i - 1].tick_params(axis = 'x', labelsize = 8)
        axes[i - 1].tick_params(axis = 'y', labelsize = 8)
        
        axes[i - 1].plot([1e-60, 1e15], [1e-60, 1e15], 'k-', lw = 0.8)
        axes[i - 1].scatter(GCMC, ML, alpha = .4, marker = '.')
        axes[i - 1].plot([1e-15, 1e-15], [1e-60, 1e15], 'k--', lw = 0.5)
        axes[i - 1].plot([1e-60, 1e15], [1e-15, 1e-15], 'k--', lw = 0.5)
        
        axes[i - 1].set_xlabel('$K_H$ from GCMC [mol/kg/Pa]', fontsize = 12)
        axes[i - 1].set_ylabel('$K_H$ from ML [mol/kg/Pa]', fontsize = 12)
        axes[i - 1].set_xscale('log')
        axes[i - 1].set_yscale('log')
        axes[i - 1].set_xlim([1e-60, 1e15])
        axes[i - 1].set_ylim([1e-60, 1e15])
        axes[i - 1].set_xticks(np.logspace(-55, 15, 8))
        axes[i - 1].set_yticks(np.logspace(-55, 15, 8))
        axes[i - 1].set_title('split #{}'.format(i))
    
    plt.tight_layout()
    plt.savefig('{}/figure3.png'.format(figure))

def figure4():
    fig, axes = plt.subplots(2, 2, figsize = (9.5, 9), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e17]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 15, 5))
        axes[i].set_yticks(np.logspace(-5, 15, 5))
        axes[i].set_xlabel('$S_{^{A}/_{B}}$ by GCMC', fontsize = 'large')
        axes[i].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
        axes[i].plot(space, space, 'k-', lw = 1)

    letter = ['a', 'b', 'c', 'd']
    sel = pd.read_csv('{}/selectivity_valid_300.tsv'.format(TSV), sep = '\t')
    #S = [0.900, 0.838, 0.392, 0.404]

    for j, pair in enumerate([('propionitrile', '1,5-heptadiene'), ('methyl isopropyl ether', 'neopentane'), ('ethylamine', 'dimethylamine'), ('propene', 'propane')]):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            df = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = df[df.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_1 = df[df.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]

            mol_2_index = df[df.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_2 = df[df.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            
            join = df_1.join(df_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_1 / join.predict_2
            join['S_GCMC'] = join.K_1 / join.K_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} / {}'.format(mol_1, mol_2))
            
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e15, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figure4.png'.format(figure), bbox_extra_artists = (ldg,), bbox_inches = 'tight')

def figure5():
    K = np.log10(train_valid.K)
    H = train_valid.H

    fig, ax = plt.subplots(figsize = (5, 4.5), dpi = 150)

    ax.scatter(K, H, marker = '.', alpha = .4)
    ax.set_xlabel('log($K_H$)')
    ax.set_ylabel('$\Delta H_{ads}$ [kJ/mol]')

    plt.tight_layout()
    plt.savefig('{}/figure5.png'.format(figure))

def figure6():
    fig, axes = plt.subplots(2, 2, figsize = (9.5, 9), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e17]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 15, 5))
        axes[i].set_yticks(np.logspace(-5, 15, 5))
        axes[i].set_xlabel('$S_{^{A}/_{B}}^{373}$ by GCMC', fontsize = 'large')
        axes[i].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
        axes[i].plot(space, space, 'k-', lw = 1)

    letter = ['a', 'b', 'c', 'd']
    sel = pd.read_csv('{}/selectivity_valid_373.tsv'.format(TSV), sep = '\t')
    #S = [0.900, 0.838, 0.392, 0.404]

    for j, pair in enumerate([('propionitrile', '1,5-heptadiene'), ('methyl isopropyl ether', 'neopentane'), ('ethylamine', 'dimethylamine'), ('propene', 'propane')]):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            K = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
            H = pd.read_csv('{}/{}/valid_H.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = K[K.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_1 = K[K.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]
            H_1 = H[H.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['H', 'predict']]

            mol_2_index = K[K.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_2 = K[K.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            H_2 = H[H.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['H', 'predict']]

            K_1['K_373'] = henry_at_diff_temp(K_1.K, H_1.H, 300, 373)
            K_1['predict_373'] = henry_at_diff_temp(K_1.predict, H_1.predict, 300, 373)
            
            K_2['K_373'] = henry_at_diff_temp(K_2.K, H_2.H, 300, 373)
            K_2['predict_373'] = henry_at_diff_temp(K_2.predict, H_2.predict, 300, 373)
            
            join = K_1.join(K_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_373_1 / join.predict_373_2
            join['S_GCMC'] = join.K_373_1 / join.K_373_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} / {}'.format(mol_1, mol_2))
            
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e15, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figure6.png'.format(figure), bbox_extra_artists = (ldg,), bbox_inches = 'tight')

def figureS1():
    fig, axes = plt.subplots(2, 5, figsize = (20, 8), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e20]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 20, 6))
        axes[i].set_yticks(np.logspace(-5, 20, 6))
        axes[i].plot(space, space, 'k-', lw = 1)

    axes[0].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
    axes[5].set_ylabel('$S_{^{A}/_{B}}$ by ML', fontsize = 'large')
    axes[7].set_xlabel('$S_{^{A}/_{B}}$ by GCMC', fontsize = 'large')

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    sel = pd.read_csv('{}/selectivity_valid_300.tsv'.format(TSV), sep = '\t')

    for j, pair in enumerate(azeo_pairs_train_valid):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            df = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = df[df.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_1 = df[df.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]

            mol_2_index = df[df.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            df_2 = df[df.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]

            join = df_1.join(df_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_1 / join.predict_2
            join['S_GCMC'] = join.K_1 / join.K_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} - {}'.format(mol_1, mol_2))
                
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]

        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e18, text, fontsize = 'large')

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figureS1.png'.format(figure), bbox_extra_artists = (ldg,), bbox_inches = 'tight')

def figureS2():
    fig, axes = plt.subplots(2, 5, figsize = (20, 8), dpi = 150)
    axes = axes.ravel()

    space = [1e-9, 1e20]

    for i in range(len(axes)):
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(space)
        axes[i].set_ylim(space)
        axes[i].set_xticks(np.logspace(-5, 20, 6))
        axes[i].set_yticks(np.logspace(-5, 20, 6))
        axes[i].plot(space, space, 'k-', lw = 1)

    axes[0].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
    axes[5].set_ylabel('$S_{^{A}/_{B}}^{373}$ by ML', fontsize = 'large')
    axes[7].set_xlabel('$S_{^{A}/_{B}}^{373}$ by GCMC', fontsize = 'large')

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    sel = pd.read_csv('{}/selectivity_valid_373.tsv'.format(TSV), sep = '\t')

    for j, pair in enumerate(azeo_pairs_train_valid):
        mol_1, mol_2 = pair

        for i in range(1, seed):
            K = pd.read_csv('{}/{}/valid_K.tsv'.format(TSV, i), sep = '\t')
            H = pd.read_csv('{}/{}/valid_H.tsv'.format(TSV, i), sep = '\t')

            mol_1_index = K[K.molecule == mol_1].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_1 = K[K.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['K', 'predict']]
            H_1 = H[H.molecule == mol_1].loc[mol_1_index].set_index('MOF')[['H', 'predict']]

            mol_2_index = K[K.molecule == mol_2].set_index('MOF')[['K', 'predict']].applymap(lambda x: x > 1e-15).all(axis = 1).values
            K_2 = K[K.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['K', 'predict']]
            H_2 = H[H.molecule == mol_2].loc[mol_2_index].set_index('MOF')[['H', 'predict']]

            K_1['K_373'] = henry_at_diff_temp(K_1.K, H_1.H, 300, 373)
            K_1['predict_373'] = henry_at_diff_temp(K_1.predict, H_1.predict, 300, 373)
            
            K_2['K_373'] = henry_at_diff_temp(K_2.K, H_2.H, 300, 373)
            K_2['predict_373'] = henry_at_diff_temp(K_2.predict, H_2.predict, 300, 373)
            
            join = K_1.join(K_2, on = ['MOF'], how = 'inner', lsuffix = '_1', rsuffix = '_2')

            join['S_ML'] = join.predict_373_1 / join.predict_373_2
            join['S_GCMC'] = join.K_373_1 / join.K_373_2

            axes[j].scatter(join.S_GCMC, join.S_ML, marker = '.', alpha = .7, label = 'split #{}'.format(i))
            axes[j].set_title('{} - {}'.format(mol_1, mol_2))
         
        S = sel[np.logical_and(sel.mol_1 == mol_1, sel.mol_2 == mol_2)]['mean'].values[0]
        
        text = '({})'.format(letter[j]) + ' S' + ' = {:.3f}'.format(S)
        axes[j].text(1e-8, 1e18, text, fontsize = 'large')
    
    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles = handles, labels = labels, loc = 'center', ncol = 5, bbox_to_anchor = (0.5, -0.02), frameon = True, bbox_transform = fig.transFigure)
    plt.tight_layout()
    plt.savefig('{}/figureS2.png'.format(figure), bbox_extra_artists = (ldg,), bbox_inches = 'tight')

figure3()
figure4()
figure5()
figure6()
figureS1()
figureS2()
