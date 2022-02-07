import numpy as np
import pandas as pd
import os
import sys

import warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.utils import shuffle

# define functions
def train_model(data, test_fold, compare):
    print('Training on {} prediction\n'.format(compare))

    model = GradientBoostingRegressor(learning_rate = 0.1, loss = 'ls')

    param_grid = {'n_estimators': [1000, 5000],
    'max_depth': [3, 4, 5]}

    ps = PredefinedSplit(test_fold)
    scorer = make_scorer(mean_absolute_error, greater_is_better = False)
    gcv = GridSearchCV(model, param_grid, cv = ps.split(), n_jobs = -1, scoring = scorer, refit = True)

    if compare == 'K':
        gcv.fit(data[all_des].values, np.log10(data.K))
    elif compare == 'H':
        gcv.fit(data[all_des].values, data.H)

    best_model = gcv.best_estimator_
    
    print('Optimal hyperparameters: {}\n'.format(gcv.best_params_))

    return best_model

# make direcotries
figure = 'figures'
if not os.path.isdir(figure):
    os.makedirs(figure)

data = 'data'
if not os.path.isdir(data):
    os.makedirs(data)

# read in the set 1
set_1 = pd.read_csv(sys.argv[1])

# read in the set 2
set_2 = pd.read_csv(sys.argv[2])
k_2 = np.log10(set_2.K)
h_2 = set_2.H

mofs = set(set_1.MOF)
molecules_1 = set(set_1.molecule)
molecules_2 = set(set_2.molecule)

# descriptors
text = set_1.columns[2:5]
pes = set_1.columns[5:33]
aprdf = set_1.columns[33:101]
mol = set_1.columns[101:137]
all_des = set_1.columns[2:137]

# ensemble modeling on set 1
seed = 11
random_seed = np.arange(1, seed * 2 - 1, 2)

ensemble_K = []
ensemble_H = []

for i, seed in enumerate(random_seed):
    print('split #{} starts'.format(i + 1))

    # make subdirecotries
    data_split = '{}/{}'.format(data, i + 1)

    if not os.path.isdir(data_split):
        os.makedirs(data_split)

    # split dataset
    train_valid, test = train_test_split(set_1, test_size = .2, stratify = set_1.molecule, random_state = seed)

    k_train_valid = np.log10(train_valid.K)
    k_test = np.log10(test.K)

    h_train_valid = train_valid.H
    h_test = test.H

    train, valid = train_test_split(train_valid, test_size = .2, stratify = train_valid.molecule, random_state = seed)

    test_fold = np.zeros(train_valid.shape[0])
    for j in train.index:
        a = train_valid.index.get_loc(j)
        test_fold[a] = -1

    # predict K
    best_model_K = train_model(train_valid, test_fold, compare = 'K')

    k_train_valid_predict = pd.Series(best_model_K.predict(train_valid[all_des].values), index = k_train_valid.index)
    k_test_predict = pd.Series(best_model_K.predict(test[all_des].values), index = k_test.index)

    k_train_valid_csv = train_valid[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_train_valid_predict, columns = ['predict']))
    k_train_valid_csv.to_csv('{}/{}.tsv'.format(data_split, 'train_valid_K'), sep = '\t', index = False)

    k_test_csv = test[['MOF', 'molecule', 'K']].join(pd.DataFrame(10.**k_test_predict, columns = ['predict']))
    k_test_csv.to_csv('{}/{}.tsv'.format(data_split, 'test_K'), sep = '\t', index = False)

    # predict H
    best_model_H = train_model(train_valid, test_fold, compare = 'H')

    h_train_valid_predict = pd.Series(best_model_H.predict(train_valid[all_des].values), index = h_train_valid.index)
    h_test_predict = pd.Series(best_model_H.predict(test[all_des].values), index = h_test.index)

    h_train_valid_csv = train_valid[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_train_valid_predict, columns = ['predict']))
    h_train_valid_csv.to_csv('{}/{}.tsv'.format(data_split, 'train_valid_H'), sep = '\t', index = False)

    h_test_csv = test[['MOF', 'molecule', 'H']].join(pd.DataFrame(h_test_predict, columns = ['predict']))
    h_test_csv.to_csv('{}/{}.tsv'.format(data_split, 'test_H'), sep = '\t', index = False)

    # gather predicted values on the set 2
    k_predict = best_model_K.predict(set_2[all_des].values)
    h_predict = best_model_H.predict(set_2[all_des].values)

    ensemble_K.append(k_predict)
    ensemble_H.append(h_predict)

    print('split #{} ended\n'.format(i + 1))

# average on the set 2
## K prediction
ensemble_K = np.array(ensemble_K)
k_predict_avg = np.mean(ensemble_K, axis = 0)
k_predict = set_2[['MOF', 'molecule', 'K']].join(pd.DataFrame(ensemble_K.T, columns = np.arange(1, len(random_seed) + 1, 1), index = k_2.index))
k_predict['predict'] = 10.**k_predict_avg
k_predict.to_csv('{}/{}.tsv'.format(data, 'set_2_K'), sep = '\t', index = False)

## H prediction
ensemble_H = np.array(ensemble_H)
h_predict_avg = np.mean(ensemble_H, axis = 0)
h_predict = set_2[['MOF', 'molecule', 'H']].join(pd.DataFrame(ensemble_H.T, columns = np.arange(1, len(random_seed) + 1, 1), index = h_2.index))
h_predict['average'] = h_predict_avg
h_predict.to_csv('{}/{}.tsv'.format(data, 'set_2_H'), sep = '\t', index = False)
