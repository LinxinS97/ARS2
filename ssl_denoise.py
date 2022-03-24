import argparse
import json
import logging
import os
import pprint

import torch
import numpy as np

from wrench import efficient_training
from wrench.classification import Denoise
from wrench.dataset import load_dataset
from wrench.evaluation import METRIC, AverageMeter
from wrench.logging import LoggingHandler
from wrench.search import grid_search

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

target_dict = {
    'f1_binary': ['sms', 'census', 'spouse', 'cdr', 'basketball', 'tennis', 'commercial'],
    'f1_macro'      : ['semeval', 'chemprot', 'agnews', 'imdb', 'trec', 'yelp', 'youtube'],
}

data_to_target = {data: metric for metric, datasets in target_dict.items() for data in datasets}

datasets = ['spouse', 'chemprot', 'agnews', 'imdb', 'sms', 'trec', 'yelp', 'youtube', 'census']

snorkel_paras = {
    'agnews'  : {'l2': 0.01, 'lr': 0.01, 'n_epochs': 200, 'seed': 32514},
    'census'  : {'l2': 0.0001, 'lr': 0.1, 'n_epochs': 10, 'seed': 20416},
    'chemprot': {'l2': 0.001, 'lr': 0.001, 'n_epochs': 5, 'seed': 32514},
    'imdb'    : {'l2': 0.001, 'lr': 0.1, 'n_epochs': 200, 'seed': 32514},
    'sms'     : {'l2': 0.0001, 'lr': 1e-05, 'n_epochs': 5, 'seed': 32514},
    'spouse'  : {'l2': 0.1, 'lr': 0.001, 'n_epochs': 5, 'seed': 32514},
    'trec'    : {'l2': 0.1, 'lr': 0.001, 'n_epochs': 5, 'seed': 20416},
    'yelp'    : {'l2': 1e-05, 'lr': 0.1, 'n_epochs': 100, 'seed': 20415},
    'youtube' : {'l2': 0.01, 'lr': 0.01, 'n_epochs': 200, 'seed': 32514}
}

search_space = {
    'optimizer_lr'          : [1e-5, 3e-5, 1e-6, 3e-6],
    'optimizer_weight_decay': [0.0],
    'alpha'                 : [0.6],
    'hidden_size'           : [64, 128, 256, 512],
    'c1'                    : [0.1, 0.3, 0.5, 0.7, 0.9],
    'c2'                    : [0.1, 0.3, 0.5, 0.7, 0.9],
}
# min_trials = 200
# patience = 200
# if_mlp = False
# default_parameter = {
#     'real_batch_size'         : 512,
#     'test_batch_size'         : 1024,
#     'n_steps'                 : 100000,
#     'grad_norm'               : -1,
#     'use_lr_scheduler'        : True,
#     'binary_mode'             : False,
#
#     'lr_scheduler'            : 'default',
#     'optimizer'               : 'default',
#
#     'backbone'                : 'MLP',
#     'backbone_n_hidden_layers': 2,
# }
min_trials = 15
patience = 20
if_mlp = False
default_parameter = {
    'batch_size'               : 16,
    'real_batch_size'          : 16,
    'test_batch_size'          : 128,
    'n_steps'                  : 10000,
    'grad_norm'                : -1,
    'use_lr_scheduler'         : True,
    'binary_mode'              : False,

    'lr_scheduler'             : 'default',
    'optimizer'                : 'default',

    'backbone'                 : 'BERT',
    'backbone_model_name'      : 'roberta-base',
    'backbone_max_tokens'      : 128,
    'backbone_fine_tune_layers': 4,
}
def filter_fn_for_denoise(grids, para_names):
    c1, c2 = para_names.index('c1'), para_names.index('c2')
    return [grid for grid in grids if grid[c1] + grid[c2] < 1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--data", type=str, default='yelp')
    parser.add_argument("--label_model", type=str, default='Snorkel')
    parser.add_argument("--imbalance", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.gpu}')


    evaluation_step = 5
    tolerance = -1
    imbalance_ratio = args.imbalance

    study_patience = 0.1
    prune_threshold = 0.1

    path = args.path
    save_dir = f'{path}/ssl/denoise'
    if if_mlp:
        save_dir = f'{save_dir}_mlp'
    os.makedirs(save_dir, exist_ok=True)
    if args.suffix == '':
        suffix = args.data
    else:
        suffix = args.suffix

    label_model_name = args.label_model or 'none'
    assert label_model_name != 'none'

    results = {}
    for data in datasets:

        if args.data != '' and data != args.data:
            continue

        target = data_to_target[data]
        if data == 'trec' and not if_mlp:
            patience = 40

        if label_model_name == 'MajorityVoting':
            default_parameter['label_model'] = 'MajorityVoting'
        elif label_model_name == 'Snorkel':
            default_parameter['label_model'] = 'Snorkel'
            default_parameter['label_model_l2'] = snorkel_paras[data]['l2']
            default_parameter['label_model_lr'] = snorkel_paras[data]['lr']
            default_parameter['label_model_n_epochs'] = snorkel_paras[data]['n_epochs']
            default_parameter['label_model_seed'] = snorkel_paras[data]['seed']

        logger.info('=' * 100)
        logger.info(f'[data]: {data}')
        logger.info(f'[label model]: {label_model_name}')
        dataset_path = f'{path}/datasets/'
        if data in ['census', 'basketball', 'tennis', 'commercial']:
            extract_fn = None
            train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=True, extract_fn=extract_fn, cache_name=extract_fn)
        else:
            extract_fn = 'bert'
            model_name = 'roberta-base'
            train_data, valid_data, test_data = load_dataset(dataset_path, data,
                                                             extract_feature=True,
                                                             extract_fn=extract_fn,
                                                             cache_name='roberta',
                                                             model_name=model_name)
            aggregated_hard_labels = np.load(f'./im_examples/label_model_output/'
                                             f'pred_imbalance{imbalance_ratio}_{data}_hard.npy')
            train_ids = np.load(f'./im_examples/label_model_output/train_ids_imbalance{imbalance_ratio}_{data}.npy')
            valid_ids = np.load(f'./im_examples/label_model_output/valid_ids_imbalance{imbalance_ratio}_{data}.npy')
            train_data = train_data.create_subset(train_ids)
            valid_data = valid_data.create_subset(valid_ids)

        if args.test:
            searched_paras = json.load(open(f'{save_dir}/denoise_{label_model_name}_{suffix}.json', 'r'))[data]['paras']
            save_dir = f'{save_dir}_test'
            os.makedirs(save_dir, exist_ok=True)
        else:
            with efficient_training(amp=True):
                searched_paras = grid_search(
                    Denoise(**default_parameter),

                    search_space=search_space,
                    filter_fn=filter_fn_for_denoise,
                    dataset_train=train_data,
                    dataset_valid=valid_data,
                    y_train=aggregated_hard_labels,

                    metric=target,
                    direction='maximize',
                    patience=patience,
                    evaluation_step=evaluation_step,
                    tolerance=tolerance,
                    valid_mode='ensemble',

                    n_repeats=args.n_repeats,
                    n_trials=args.n_trials,
                    min_trials=min_trials,
                    study_patience=study_patience,
                    prune_threshold=prune_threshold,
                    study_timeout=300 * 60 * 60,

                    parallel=if_mlp,
                    device=device,
                )
        pprint.pprint(searched_paras)

        log = {}
        meter = AverageMeter(names=METRIC.keys())
        for i in range(5):
            model = Denoise(**default_parameter, **searched_paras)
            with efficient_training(amp=not if_mlp):
                history = model.fit(
                    dataset_train=train_data,
                    y_train=aggregated_hard_labels,
                    dataset_valid=valid_data,
                    device=device,
                    metric=target,
                    direction='maximize',
                    patience=patience,
                    valid_mode='ensemble',
                    evaluation_step=evaluation_step,
                )
            proba_y = model.predict_proba(test_data)
            metrics = {metric: metric_fn(test_data.labels, proba_y) for metric, metric_fn in METRIC.items()}
            log[i] = {
                'metrics': metrics,
                'history': history,
            }
            meter.update(**metrics)

        metrics = meter.get_results()
        pprint.pprint(metrics)

        best_log = {
            'metrics': metrics,
            'exp'    : log,
            'paras'  : searched_paras}
        results[data] = best_log

    json.dump(results, open(f'{save_dir}/denoise_{label_model_name}_{suffix}.json', 'w'), indent=4)
