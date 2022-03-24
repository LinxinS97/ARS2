import argparse
import json
import logging
import os
import pprint

import torch

from wrench.dataset import load_dataset
from wrench.classification import DDSelfTrain, LDSelfTrain
from wrench.evaluation import METRIC, AverageMeter
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench import efficient_training

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

target_dict = {
    'f1_binary': ['sms', 'census', 'spouse', 'cdr', 'basketball', 'tennis', 'commercial'],
    'acc'      : ['semeval', 'chemprot', 'agnews', 'imdb', 'trec', 'yelp', 'youtube'],
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

# search_space = {
#     'optimizer_lr'          : [1e-5, 3e-5, 5e-5],
#     'optimizer_weight_decay': [1e-4],
#     'label_model_update'        : [50, 100, 200],
#     'filter_mode'                 : ['thres'],
#     'filter_theta'                : [0.1, 0.3, 0.5, 0.7, 0.9],
# }
# min_trials = 100
# patience = 20
# if_mlp = False
# default_parameter = {
#     'batch_size'               : 32,
#     'real_batch_size'          : 8,
#     'test_batch_size'          : 64,
#     'n_steps'                  : 10000,
#     'grad_norm'                : -1,
#     'use_lr_scheduler'         : True,
#     'binary_mode'              : False,
#
#     'lr_scheduler'             : 'default',
#     'optimizer'                : 'default',
#
#     'backbone'                 : 'BERT',
#     'backbone_model_name'      : 'roberta-base',
#     'backbone_max_tokens'      : 512,
#     'backbone_fine_tune_layers': -1,
# }

search_space = {
    'optimizer_lr'          : [1e-3, 1e-4, 1e-2],
    'optimizer_weight_decay': [0.0],
    'dropout'               : [0.3],
    'batch_size'            : [32, 128, 512],

    'backbone_dropout'      : [0.2, 0.0],
    'backbone_hidden_size'  : [128, 256, 512],

    'label_model_update'        : [50, 100, 200],
    'filter_mode'                 : ['thres'],
    'filter_theta'                : [0.1, 0.3, 0.5, 0.7, 0.9],
}
min_trials = 200
patience = 200
if_mlp = True
default_parameter = {
    'real_batch_size'         : 512,
    'test_batch_size'         : 1024,
    'n_steps'                 : 100000,
    'grad_norm'                : -1,
    'use_lr_scheduler'         : True,
    'binary_mode'              : False,

    'lr_scheduler'             : 'default',
    'optimizer'                : 'default',

    'backbone'                : 'MLP',
    'backbone_n_hidden_layers': 2,
}

def filter_fn_for_selftrain(grids, para_names):
    mode_id = para_names.index('filter_stage')
    thres_id = para_names.index('filter_theta')
    new_grids = []
    for g in grids:
        if g[mode_id] == 'none':
            g = list(g)
            g[thres_id] = 0.1
            g = tuple(g)
        if g not in new_grids:
            new_grids.append(g)
    return new_grids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='../')
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--data", type=str, default='youtube')
    parser.add_argument("--model", type=str, default='ddselftrain', choices=['ddselftrain', 'ldselftrain'])
    parser.add_argument("--filter_stage", type=str, default='none', choices=['both' , 'first' , 'none'])
    parser.add_argument("--label_model", type=str, default='Snorkel')
    parser.add_argument("--soft_label", type=int, default=1)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=4, help="id of gpu")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print(args)

    soft_label = args.soft_label
    filter_stage = args.filter_stage
    search_space['filter_stage'] = [filter_stage]

    if args.model == 'ddselftrain':
        self_train_model = DDSelfTrain
    elif args.model == 'ldselftrain':
        self_train_model = LDSelfTrain
    else:
        raise ValueError(f'unknown {args.model}')

    device = torch.device(f'cuda:{args.gpu}')

    evaluation_step = 5
    tolerance = -1

    study_patience = 0.1
    prune_threshold = 0.1

    path = args.path
    if soft_label:
        save_dir = f'{path}/ssl/{args.model}_soft_{filter_stage}'
    else:
        save_dir = f'{path}/ssl/{args.model}_hard_{filter_stage}'
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
        if data == 'trec' and not if_mlp:
            patience = 40

        target = data_to_target[data]

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
        if if_mlp:
            if data in ['census', 'basketball', 'tennis', 'commercial']:
                extract_fn = None
                train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=True, extract_fn=extract_fn, cache_name=extract_fn)
            else:
                extract_fn = 'bert'
                model_name = 'roberta-base'
                train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=True, extract_fn=extract_fn, cache_name='roberta', model_name=model_name)
        else:
            train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)

        if args.test:
            searched_paras = json.load(open(f'{save_dir}/{args.model}_{label_model_name}_{suffix}.json', 'r'))[data]['paras']
            save_dir = f'{save_dir}_test'
            os.makedirs(save_dir, exist_ok=True)
        else:
            with efficient_training(amp=True):
                searched_paras = grid_search(
                    self_train_model(**default_parameter),

                    search_space=search_space,
                    filter_fn=filter_fn_for_selftrain,
                    dataset_train=train_data,
                    dataset_valid=valid_data,
                    soft_labels=soft_label,

                    metric=target,
                    direction='maximize',
                    patience=patience,
                    evaluation_step=evaluation_step,
                    tolerance=tolerance,

                    n_repeats=args.n_repeats,
                    n_trials=args.n_trials,
                    min_trials=min_trials,
                    study_patience=study_patience,
                    prune_threshold=prune_threshold,
                    study_timeout=300*60*60,

                    parallel=if_mlp,
                    device=device,
                )
        pprint.pprint(searched_paras)

        log = {}
        meter = AverageMeter(names=METRIC.keys())
        for i in range(5):
            model = self_train_model(**default_parameter, **searched_paras)
            with efficient_training(amp=not if_mlp):
                history = model.fit(
                    dataset_train=train_data,
                    dataset_valid=test_data,
                    soft_labels=soft_label,
                    device=device,
                    metric=target,
                    direction='maximize',
                    patience=patience,
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

    json.dump(results, open(f'{save_dir}/{args.model}_{label_model_name}_{suffix}.json', 'w'), indent=4)
