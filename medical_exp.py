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


This script is used to run a comparative study on the given dataset
Run python3 medical_exp.py --help to access the help on how to use this script.
"""

import os.path
import numpy as np

import argparse
import pickle
import utils


available_datasets = [
    'COVID',
    'MYOCARDIAL',
    'NHANES',
]

imputation_models = [
    'MISSFOREST',
    'SOFTIMPUTE',
    'GAIN',
    'MIDA',
    'SINKHORN',
]

evaluation_metrics = [
    'bal_acc',
    'auc_ovo',
    'f1',
]

approaches = [
    'SINGLE',
    'MULTIPLE',
    'SINGLE-HOT-PATCH',
    'HOT-PATCH',
]


def run_expe(metrics, missing_mask, data_missing, y, imputation_model, n_runs,
             n_imputations, epochs, n_batches, hidden_layer_sizes, alpha, imp_name, exp_name):
    # Execute n imputations and get results
    data_missing_nans = np.where(missing_mask, data_missing, np.nan)
    imputations = utils.run_imputations(data_missing_nans, missing_mask, data_missing, imputation_model, n_imputations, imp_name)
    # Run experiments n_runs times
    # All are then evaluated in the same way
    for approach in approaches:
        # Do not run if already executed and saved in metrics
        if not metrics[imputation_model][approach][evaluation_metrics[0]]:
            np.random.seed(0)
            for run in range(n_runs):
                y_probas, y_test = utils.run_approach(approach, imputations, y, run, n_runs, epochs, n_batches, hidden_layer_sizes, alpha)
                res = utils.eval_results(y_probas, y_test, evaluation_metrics)
                info = f'  {approach} RUN {run+1:03d}/{n_runs:03d}'
                for m, v in res.items():
                    metrics[imputation_model][approach][m].append(v)
                    info += f' - {m}: {v:.4f}%'
                print(info, flush=True)
            info = f'GLOBAL {imp_name} {approach}'
            for metric in evaluation_metrics:
                info += f' - {metric}: {np.mean(metrics[imputation_model][approach][metric]):.4f}% +- '
                info += f'{np.std(metrics[imputation_model][approach][metric]):.4f}%'
            print(info, flush=True)
            # Save metrics to pickle after each end of experiment to get regular savings
            if not os.path.exists('results'):
                os.makedirs('results')
            pickle.dump(metrics, open(f'results/{exp_name}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def main(dataset, n_runs, n_imputations, epochs, n_batches, hidden_layer_sizes, alpha):
    exp_name = f'{dataset}_n{n_imputations}_r{n_runs}_e{epochs}_b{n_batches}' + \
               f'_i{"_".join(map(str, hidden_layer_sizes))}_a{str(alpha).replace(".", "_")}'
    if os.path.isfile(f'results/{exp_name}.pickle'):
        metrics = pickle.load(open(f'results/{exp_name}.pickle', 'rb'))
        imputation_models_to_exec = []
        for imputation_model in imputation_models:
            exec_model = False
            if not imputation_model in metrics:
                exec_model = True
                metrics[imputation_model] = {
                    approach: {metric: [] for metric in evaluation_metrics} for approach in approaches
                }
            else:
                for approach in approaches:
                    if approach not in metrics[imputation_model]:
                        exec_model = True
                        metrics[imputation_model][approach] = {metric: [] for metric in evaluation_metrics}
            if exec_model:
                imputation_models_to_exec.append(imputation_model)
        if not imputation_models_to_exec:
            print(f'## ' + '-' * len(f'Skip {exp_name}') + ' ##', flush=True)
            print(f'## Skip {exp_name} ##', flush=True)
            print(f'## ' + '-' * len(f'Skip {exp_name}') + ' ##', flush=True)
            return
    else:
        # Dictionary to collect results
        metrics = {
            imputation_model: {
                approach: {metric: [] for metric in evaluation_metrics} for approach in approaches
            } for imputation_model in imputation_models
        }
        imputation_models_to_exec = imputation_models
    
    # Load real-world dataset containing natural missing values
    if dataset == 'COVID':
        data_missing, missing_mask, y = utils.load_covid()
    if dataset == 'MYOCARDIAL':
        data_missing, missing_mask, y = utils.load_myocardial()
    if dataset == 'NHANES':
        data_missing, missing_mask, y = utils.load_nhanes()
    np.random.seed(0)
    p = np.random.permutation(len(data_missing))
    data_missing, missing_mask, y = data_missing[p], missing_mask[p], y[p]
    
    # Run expe using each imputation model
    for imputation_model in imputation_models_to_exec:
        # Run expe
        print(f'## ' + '-' * len(f'MODEL {imputation_model}') + ' ##', flush=True)
        print(f'## MODEL {imputation_model} ##', flush=True)
        imp_name = f'{dataset}_m{imputation_model}'
        run_expe(metrics, missing_mask, data_missing, y, imputation_model, n_runs,
                 n_imputations, epochs, n_batches, hidden_layer_sizes, alpha, imp_name, exp_name)
        print(f'## ' + '-' * len(f'MODEL {imputation_model}') + ' ##\n', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=available_datasets,
                        default='COVID', help='name of the dataset to use, default is COVID')
    parser.add_argument('-r', '--n-runs', type=int,
                        default=200, help='number of runs, default is 200')
    parser.add_argument('-n', '--n-imputations', type=int,
                        default=20, help='number of imputations to run, default is 20')
    parser.add_argument('-e', '--epochs', type=int,
                        default=50, help='number of epochs, default is 50')
    parser.add_argument('-b', '--n-batches', type=int,
                        default=10, help='number of batches, default is 10')
    parser.add_argument('-i', '--hidden-layer-sizes', nargs='+', type=int,
                        default=[32, 32], help='MLP hidden layer sizes, default is [32, 32]')
    parser.add_argument('-q', '--alpha', type=float,
                        default=1., help='scale to apply to the stds (alpha * stds), default is 1.')
    args = vars(parser.parse_args())
    
    main(**args)
