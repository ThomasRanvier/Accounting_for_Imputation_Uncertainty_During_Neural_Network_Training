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


This script is used to run MICE on a given dataset
Run python3 mice_exp.py --help to access the help on how to use this script.
"""

import os.path
import numpy as np
from utils_na import inject_missing

import argparse
import pickle
import utils

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


available_datasets = [
    'IRIS',
    'AUSTRALIAN',
    'WINE',
    'ABALONE',
    'PIMA',
]

missing_modes = [
    'MCAR',
    'MAR',
    'MNAR',
]

missing_rates = [
    .1,
    .15,
    .25,
]

evaluation_metrics = [
    'bal_acc',
    'auc_ovo',
    'f1',
]


def main(dataset, n_runs, epochs, n_batches, hidden_layer_sizes):
    exp_name = f'MICE_{dataset}_r{n_runs}_e{epochs}'
    if os.path.isfile(f'results/{exp_name}.pickle'):
        print(f'## ' + '-' * len(f'Skip {exp_name}') + ' ##', flush=True)
        print(f'## Skip {exp_name} ##', flush=True)
        print(f'## ' + '-' * len(f'Skip {exp_name}') + ' ##', flush=True)
        return
    # Load dataset
    x, y = utils.load_ds(dataset, shuffle_seed=0)
    # Dictionary to collect results
    metrics = {missing_mode: {missing_rate: {m: []
                                             for m in evaluation_metrics}
                              for missing_rate in missing_rates}
               for missing_mode in missing_modes}
    # For each missing values setting run expe
    for missing_mode in missing_modes:
        for missing_rate in missing_rates:
            print(f'## ' + '-' * len(f'MICE {missing_mode} {missing_rate}%') + ' ##', flush=True)
            print(f'## MICE {missing_mode} {missing_rate}% ##', flush=True)
            # Inject missing values given missing mode and rate
            data_missing, missing_mask = inject_missing(x, missing_rate, missing_mode, random_seed=0)
            data_missing_nans = np.where(missing_mask, data_missing, np.nan)
            # Get MICE imputation
            imputer = IterativeImputer(random_state=0)
            imputation = imputer.fit_transform(data_missing_nans)
            # Run MLP train+test 200x on imputation
            np.random.seed(0)
            for run in range(n_runs):
                y_probas, y_test = utils.run_approach('SINGLE', [imputation], y, run, n_runs,
                                                epochs, n_batches, hidden_layer_sizes)
                res = utils.eval_results(y_probas, y_test, evaluation_metrics)
                info = f'  RUN {run+1:03d}/{n_runs:03d}'
                for m, v in res.items():
                    metrics[missing_mode][missing_rate][m].append(v)
                    info += f' - {m}: {v:.4f}%'
                print(info, flush=True)
            info = f'GLOBAL '
            for metric in evaluation_metrics:
                info += f' - {metric}: {np.mean(metrics[missing_mode][missing_rate][metric]):.4f}% +- '
                info += f'{np.std(metrics[missing_mode][missing_rate][metric]):.4f}%'
            print(info, flush=True)
            print(f'## ' + '-' * len(f'MICE {missing_mode} {missing_rate}%') + ' ##', flush=True)
    # Save metrics to pickle
    if not os.path.exists('results'):
        os.makedirs('results')
    pickle.dump(metrics, open(f'results/{exp_name}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets,
                        default='IRIS', help='name of the dataset to use, default is IRIS')
    parser.add_argument('-r', '--n-runs', type=int,
                        default=200, help='number of runs, default is 200')
    parser.add_argument('-e', '--epochs', type=int,
                        default=50, help='number of epochs, default is 50')
    parser.add_argument('-b', '--n-batches', type=int,
                        default=20, help='number of batches, default is 20')
    parser.add_argument('-i', '--hidden-layer-sizes', nargs='+', type=int,
                        default=[32, 32], help='MLP hidden layer sizes, default is [32, 32]')
    args = vars(parser.parse_args())
    
    main(**args)