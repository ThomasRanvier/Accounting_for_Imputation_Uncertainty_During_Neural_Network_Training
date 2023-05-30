"""
MIT License

Accounting for Imputation Uncertainty During Neural Network Training

Copyright (c) 2022 Thomas RANVIER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
## Disable tf future deprecated messages
import logging
logging.getLogger('tensorflow').disabled = True
## Disable tf CUDA messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import sklearn.datasets
import pickle
import os.path
import sys
import pandas as pd
import scipy.io

import sklearn.neighbors
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_even_slices

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from models.soft_impute import SoftImpute
from models.sinkhorn import OTimputer
from models.gain import gain
from models.mida import mida


def run_imputations(data_missing_nans, missing_mask, data_missing, imputation_model, n_imputations, imp_name=None):
    if imp_name is not None and os.path.isfile(f'imputations/{imp_name}.pickle'):
        imputations = pickle.load(open(f'imputations/{imp_name}.pickle', 'rb'))[:n_imputations]
    else:
        imputations = []
    if len(imputations) < n_imputations:
        # Execute remaining imputations to run to reach n imputations
        for n in range(len(imputations), n_imputations):
            if imputation_model == 'MISSFOREST':
                imputer = MissForest(n_jobs=-1, random_state=n)
                imputed = imputer.fit_transform(data_missing_nans)
            elif imputation_model == 'SOFTIMPUTE':
                imputer = SoftImpute(random_state=n)
                imputer.fit(data_missing_nans)
                imputed = imputer.predict(data_missing_nans)
                imputed = np.where(missing_mask, data_missing, imputed)
            elif imputation_model == 'GAIN':
                imputed = gain(data_missing_nans, {'batch_size': 128, 'hint_rate': .9, 'alpha': 100, 'iterations': 5000})
            elif imputation_model == 'MIDA':
                num_layers = 2
                num_epochs = 2000
                imputed = mida(data_missing_nans, num_layers=num_layers, num_epochs=num_epochs)
                imputed = np.where(missing_mask, data_missing, imputed)
            elif imputation_model == 'SINKHORN':
                imputer = OTimputer(niter=1000)
                imputed = imputer.fit_transform(data_missing_nans).cpu().detach().numpy()
            imputations.append(imputed)
        imputations = np.array(imputations)
        if imp_name is not None:
            if not os.path.exists('imputations'):
                os.makedirs('imputations')
            pickle.dump(imputations, open(f'imputations/{imp_name}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return imputations


def run_approach(approach, imputations, y, run, n_runs, epochs, n_batches, hidden_layer_sizes, alpha=1.):
    if approach == 'SINGLE':
        # If single imputation approach: use current imputation as dataset
        if n_runs >= len(imputations):
            current_imputation = imputations[int(run / int(n_runs / len(imputations)))]
        else:
            current_imputation = imputations[run]
        x_train, x_test, y_train, y_test = train_test_split(current_imputation, y, test_size=.33, random_state=run)
    elif approach == 'MULTIPLE':
        # If multiple imputation approach: keep n train and test sets to train one classifier on each
        y_train, y_test = train_test_split(y, test_size=.33, random_state=run)
        x_trains, x_tests = [], []
        for imputation in imputations:
            x_train, x_test = train_test_split(imputation, test_size=.33, random_state=run)
            x_trains.append(x_train)
            x_tests.append(x_test)
    elif approach == 'HOT-PATCH':
        # If multiple hot-patching imputation approach: extract stds
        stds = np.std(imputations, axis=0)
        stds_train, stds_test, y_train, y_test = train_test_split(stds, y, test_size=.33, random_state=run)
        x_trains, x_tests = [], []
        for imputation in imputations:
            x_train, x_test = train_test_split(imputation, test_size=.33, random_state=run)
            x_trains.append(x_train)
            x_tests.append(x_test)
    elif approach == 'SINGLE-HOT-PATCH':
        # If single hot-patching imputation approach: extract means and stds
        means, stds = np.mean(imputations, axis=0), np.std(imputations, axis=0)
        x_train, x_test, stds_train, stds_test, y_train, y_test = train_test_split(means, stds, y, test_size=.33, random_state=run)
    # Initialize MLP(s)
    if approach == 'MULTIPLE' or approach == 'HOT-PATCH':
        clfs = [MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=run+i) for i in range(len(imputations))]
    else:
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=run)
    classes = np.unique(y_train)
    # Train on train set(s)
    for epoch in range(epochs):
        # Iterate through batches
        p = np.random.permutation(len(x_train))
        if approach == 'HOT-PATCH':
            # Hot-patch missing values by drawing random numbers on Gaussians given their means and stds values
            x_trains_patched = [np.random.normal(x_trains[i], stds_train * alpha) for i in range(len(imputations))]
        if approach == 'SINGLE-HOT-PATCH':
            # Hot-patch missing values by drawing random numbers on Gaussians given their means and stds values
            x_train_patched = np.random.normal(x_train, stds_train * alpha)
        for sli in gen_even_slices(len(x_train), n_batches):
            # Extract batch indices
            batch_idcs = p[sli]
            if approach == 'MULTIPLE':
                for i in range(len(imputations)):
                    # Fit model to batch
                    clfs[i].partial_fit(x_trains[i][batch_idcs], y_train[batch_idcs], classes=classes)
            elif approach == 'HOT-PATCH':
                for i in range(len(imputations)):
                    # Fit model to batch
                    clfs[i].partial_fit(x_trains_patched[i][batch_idcs], y_train[batch_idcs], classes=classes)
            else:
                # Fit model to batch
                x_train_batch = x_train_patched[batch_idcs] if approach == 'SINGLE-HOT-PATCH' else x_train[batch_idcs]
                # Fit model to batch
                clf.partial_fit(x_train_batch, y_train[batch_idcs], classes=classes)
    # Predict on test set
    if approach == 'HOT-PATCH':
        # Hot-patch missing values by drawing random numbers on Gaussians given their means and stds values
        y_probas_all = []
        for i in range(len(imputations)):
            for epoch in range(epochs):
                x_test_patched = np.random.normal(x_tests[i], stds_test * alpha)
                y_probas_all.append(clfs[i].predict_proba(x_test_patched))
        y_probas = np.mean(np.array(y_probas_all), axis=0)
    elif approach == 'SINGLE-HOT-PATCH':
        # Hot-patch missing values by drawing random numbers on Gaussians given their means and stds values
        y_probas_all = []
        for epoch in range(epochs):
            x_test_patched = np.random.normal(x_test, stds_test * alpha)
            y_probas_all.append(clf.predict_proba(x_test_patched))
        y_probas = np.mean(np.array(y_probas_all), axis=0)
    elif approach == 'MULTIPLE':
        y_probas_all = []
        for i in range(len(imputations)):
            y_probas_all.append(clfs[i].predict_proba(x_tests[i]))
        y_probas = np.mean(np.array(y_probas_all), axis=0)
    else:
        y_probas = clf.predict_proba(x_test)
    return y_probas, y_test


def eval_results(y_probas, y_test, evaluation_metrics):
    y_pred = np.argmax(y_probas, axis=-1)
    res = {}
    for metric in evaluation_metrics:
        if metric == 'bal_acc':
            res[metric] = balanced_accuracy_score(y_test, y_pred) * 100.
        if metric == 'auc_ovo':
            res[metric] = roc_auc_score(y_test, y_probas[:, 1] if y_probas.shape[1] == 2 else y_probas, multi_class='ovo') * 100.
        if metric == 'f1':
            res[metric] = f1_score(y_test, y_pred, average='weighted') * 100.
    return res


def normalize(x):
    return ((x - x.min(0)) / x.ptp(0)).astype(np.float32)


def load_ds(dataset_name, shuffle_seed=0):
    if dataset_name == 'IRIS':
        x, y = sklearn.datasets.load_iris(return_X_y=True)
        x = normalize(x)
    if dataset_name == 'AUSTRALIAN':
        data_raw = pd.read_csv(f'data/australian.data', sep=',', header=None).to_numpy().astype(np.float32)
        y = data_raw[:, -1].astype(int)
        x = normalize(data_raw[:, :-1])
    if dataset_name == 'ABALONE':
        data_raw = pd.read_csv(f'data/abalone.data', sep=',', header=None)
        data_raw[[0]] = data_raw[[0]].apply(lambda col:pd.Categorical(col).codes)
        data_raw = data_raw.to_numpy().astype(np.float32)
        y = data_raw[:, -1].astype(int) - 1
        y = np.where(y <= 8, 0, 1)
        x = normalize(data_raw[:, :-1])
    if dataset_name == 'PIMA':
        data_raw = pd.read_csv(f'data/pima.csv', sep=',', header=None).to_numpy().astype(np.float32)
        y = data_raw[:, -1].astype(int)
        x = normalize(data_raw[:, :-1])
    if dataset_name == 'WINE':
        x, y = sklearn.datasets.load_wine(return_X_y=True)
        x = normalize(x)
    # Shuffle
    np.random.seed(shuffle_seed)
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    return x, y


def get_details(dataset):
    """
    Give the views names and dimensions for multi-view datasets.

    Args:
        dataset (str): The dataset name.

    Returns:
        A dictionary of the form {'view name': view dimension} and the number of elements in the dataset.
    """
    if 'NHANES' in dataset:
        views = [('df', 56), ('lab', 21), ('exam', 6), ('quest', 14)]
        num_elements = 24369
    return views, num_elements


def load_mat_dataset(dataset):
    """
    Load a .mat dataset given its name.

    Args:
        dataset (str): The dataset name.

    Returns:
        The data and the labels, both in numpy arrays, and a dictionary of the form {'view name': view dimension} and
        the number of elements in the dataset.
    """
    views, num_elements = get_details(dataset)
    mat_contents = scipy.io.loadmat(f'data/{dataset}.mat')
    y = mat_contents['y'].squeeze()
    X = [None] * len(views)
    for i in range(len(views)):
        dv = mat_contents['X'][0, i].shape[1]
        view_index = [view[1] for view in views].index(dv)
        X[view_index] = mat_contents['X'][0, i].astype(np.float32)
    return X, y, views, num_elements


def load_covid():
    data = pd.read_excel('data/covid/covid_original.xlsx', index_col=[0, 1])
    data = data.dropna(thresh=6)
    data = data.reset_index(level=1)
    n_days=1
    dropna=False
    subset=None
    time_form='diff'
    t_diff = data['出院时间'].dt.normalize() - data['RE_DATE'].dt.normalize()
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    data = data.set_index('t_diff', append=True)
    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    ).groupby('PATIENT_ID').tail(1)
    if dropna:
        data = data.dropna(subset=subset)
    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])
    ## Outcome: '出院方式'
    y = data['出院方式'].values.astype(int)
    ## Drop outcome and dates from data
    data = data.drop(columns=['入院时间', '出院时间', '出院方式'])
    data = data.apply(pd.to_numeric, errors='coerce')
    missing_mask = np.where(data.values != data.values, 0, 1)
    data_missing = data.fillna(value=data.mean()).values
    data_missing[-10, -11] = 0. ## Don't why it does not work here but no time to fix
    data_missing = ((data_missing - data_missing.min(0)) / data_missing.ptp(0)).astype(np.float32)
    data_missing = data_missing * missing_mask
    data_missing = np.concatenate((data_missing, data_missing[:, :4]), axis=1)
    missing_mask = np.concatenate((missing_mask, missing_mask[:, :4]), axis=1)
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * np.sum(~missing_mask.astype(bool)) / np.prod(data_missing.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y


def load_myocardial():
    """
    0 alive, 1 dead
    """
    df = pd.read_csv('data/miocardial_infarction/miocardial_infarction.data', sep=',', header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    y = df.iloc[:, 112:]
    y = np.where(y.to_numpy()[:, -1].astype(int) >= 1, 1, 0)
    df = df.drop([0, 7, 34, 35, 88, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123], axis=1)
    missing_mask = np.where(df.values != df.values, 0, 1)
    data_missing = df.fillna(value=df.mean()).values
    data_missing = ((data_missing - data_missing.min(0)) / data_missing.ptp(0)).astype(np.float32)
    data_missing = data_missing * missing_mask
    # 107 features, pad to 110 for convs
    data_missing = np.concatenate((data_missing[:, :3], data_missing), axis=1)
    missing_mask = np.concatenate((missing_mask[:, :3], missing_mask), axis=1)
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * np.sum(~missing_mask.astype(bool)) / np.prod(data_missing.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y


def load_nhanes():
    ## Load original dataset splitted by views
    data_raw, y, views, num_elements = load_mat_dataset('NHANES')
    ## Concat views
    data = np.concatenate(data_raw, -1)
    ## Manually select 1000 random samples from class 1 and 1000 random samples from class 2
    np.random.seed(1)
    p = np.random.permutation(len(data))
    data, y = data[p], y[p]
    class_1_indices = np.random.choice(np.where(y == 1)[0], size=1000, replace=False)
    class_2_indices = np.random.choice(np.where(y == 2)[0], size=1000, replace=False)
    data = np.concatenate([data[class_1_indices, :], data[class_2_indices, :]])
    y = np.concatenate([y[class_1_indices], y[class_2_indices]])
    np.random.seed(2)
    p = np.random.permutation(len(data))
    data, y = data[p], y[p]
    # Convert labels to binary
    y = np.where(y==2, 0, 1)
    ## Drop useless columns
    data = pd.DataFrame(data)
    for col in data.columns:
        if len(data[col].unique()) == 1:
            data.drop(col, inplace=True, axis=1)
    missing_mask = np.where(data.values != data.values, 0, 1)
    n = np.sum(~missing_mask.astype(bool))
    ## Normalize
    data = data.fillna(value=data.mean()).values
    data = ((data - data.min(0)) / data.ptp(0)).astype(np.float32)
    ## Replace missing values by 0
    data = data * missing_mask
    data_missing = data
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * n / np.prod(data.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y