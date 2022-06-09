import json
from json import JSONDecodeError

import argparse
import torch
import numpy as np
import sys
sys.path.insert(0, '/root/wrench')
from wrench import efficient_training
from wrench.dataset import load_dataset
from wrench.search import grid_search
# loss modified
from ars2 import EndClassifierModel

# BERT
default_parameter = {
    'batch_size': 16,
    'real_batch_size': 16,
    'test_batch_size': 256,
    'n_steps': 10000,
    'grad_norm': -1,
    'use_lr_scheduler': True,
    'binary_mode': False,

    'lr_scheduler': 'default',
    'optimizer': 'default',

    'backbone': 'BERT',
    'backbone_model_name': 'roberta-base',
    'backbone_max_tokens': 128,
    'backbone_fine_tune_layers': -1
}

search_space = {
    'normal': {
        'loss_type': ['normal'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'adjust_logit': [False],
        'optimizer_weight_decay': [1e-4],
    },
    'normal_la': {
        'loss_type': ['normal'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'adjust_logit': [True],
        'linear_ratio': np.linspace(1, 10),
        'score_threshold': np.linspace(-0.2, 0.2, num=20),
        'optimizer_weight_decay': [1e-4],
    },
    'en': {
        'loss_type': ['en'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'beta': np.array([0.9, 0.99, 0.999, 0.9999]),
        'gamma': np.array([0.5, 1.0, 2.0]),
        'en_type': ['softmax'],
        'adjust_logit': [False],
        'optimizer_weight_decay': [1e-4],
    },
    'en_la': {
        'loss_type': ['en'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'beta': np.array([0.9, 0.99, 0.999, 0.9999]),
        'gamma': np.array([0.5, 1.0, 2.0]),
        'en_type': ['softmax'],
        'adjust_logit': [True],
        'optimizer_weight_decay': [1e-4],
    },
    'dice': {
        'loss_type': ['dice'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'dice_smooth': np.logspace(-4, 0, num=8, base=10),
        'dice_alpha': np.arange(0.1, 1, 0.1),
        'dice_square': [True, False],
        'optimizer_weight_decay': [1e-4],
    },
    'ldam': {
        'loss_type': ['ldam'],
        'optimizer_lr': [1e-5, 3e-5, 1e-6, 3e-6],
        'max_m': np.arange(0.1, 1, 0.1),
        's': np.arange(1, 30, 3, dtype=float),
        'optimizer_weight_decay': [1e-4],
    }
}

ranking_parameter = {
    'score_type': 'margin',  # pred: P(A), margin: P(A)-P(B)
    'mean_score_type': 'mean',  # mean, taylor_mean
    're_sample_type': 'class_top',
    're_sample_concat': False,
    're_correction': True,
    'linear_ratio': 30,
    'score_threshold': 0,
    'ndcg_ratio': 0.2,
    'dev_mode': False
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--ir', type=str, help='imbalance ratio', required=True)
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu (e.g. 0)')
    args = parser.parse_args()

    data_path = args.data_path
    data = args.data
    imbalance_ratio = args.ir
    loss_type = args.loss_type

    device = torch.device(f'cuda:{args.gpu}')
    dataset_path = f'{data_path}/datasets/'
    pred_path = f'{data_path}/im_examples'

    # loss_func_list = ['normal_la', 'en', 'en_la', 'dice', 'ldam']
    n_trials = 40
    n_repeats = 1

    # imbalance_ratio = 1  # naturally imbalance dataset
    extract_fn = 'bert'
    model_name = 'roberta-base'
    train_data, valid_data, test_data = load_dataset(
        dataset_path, data,
        extract_feature=True,
        extract_fn=extract_fn,
        cache_name='roberta',
        model_name=model_name
    )
    aggregated_hard_labels = np.load(f'{pred_path}/label_model_output/pred_imbalance{imbalance_ratio}_{data}_hard.npy')

    train_ids = np.load(f'{pred_path}/label_model_output/train_ids_imbalance{imbalance_ratio}_{data}.npy')
    valid_ids = np.load(f'{pred_path}/label_model_output/valid_ids_imbalance{imbalance_ratio}_{data}.npy')

    train_data = train_data.create_subset(train_ids).get_covered_subset()
    valid_data = valid_data.create_subset(valid_ids)
    #### Run end model: MLP
    model = EndClassifierModel(**default_parameter, **ranking_parameter)

    #### Search best hyper-parameters using validation set in parallel
    searched_paras = grid_search(
        model,
        dataset_train=train_data,
        dataset_valid=valid_data,
        y_train=aggregated_hard_labels,
        patience=40,
        evaluation_step=5,
        metric='f1_macro',
        direction='auto',
        search_space=search_space[loss_type],
        n_repeats=n_repeats,
        n_trials=n_trials,
        grid=True,
        score_step=10,
        avg_score_step=20,
        distillation=True,
        device=device
    )

    try:
        with open(f"{data_path}/eval_res/optimized_roberta_res_imbalance{imbalance_ratio}_{data}.json", "r") as load_f:
            try:
                optimized_param = json.load(load_f)
            except JSONDecodeError:
                optimized_param = {}
    except FileNotFoundError:
        optimized_param = {}
    optimized_param.update({loss_type: searched_paras})
    with open(f"{data_path}/eval_res/optimized_roberta_res_imbalance{imbalance_ratio}_{data}.json", "w+") as f:
        json.dump(optimized_param, f)
