import argparse
import logging
import torch
import json
import numpy as np
import sys
sys.path.append('../')

from wrench.evaluation import AverageMeter, METRIC
from wrench import efficient_training
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
# loss modified
from ars2 import EndClassifierModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='../')
    parser.add_argument("--dataset_path", type=str, default='../datasets/')
    parser.add_argument("--data", type=str)
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--real_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--imbalance_ratio", type=int, default=10)
    parser.add_argument("--re_sample_type", type=str, default='class_top')
    parser.add_argument("--re_correction", type=bool, default=True)
    parser.add_argument("--linear_ratio", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--loss_type", type=str, default='normal_la')
    parser.add_argument("--gpuid", type=int, default=0, help="id of gpu")
    args = parser.parse_args()

    default_parameter = {
        'batch_size': args.batch_size,
        'real_batch_size': args.real_batch_size,
        'test_batch_size': args.test_batch_size,
        'n_steps': args.n_steps,
        'use_lr_scheduler': True,
        'lr_scheduler': 'default',
        'optimizer': 'default',
        'binary_mode': False,
        'backbone': 'MLP',
        'backbone_n_layers': 2,
        'backbone_hidden_size': 128
    }

    data = args.data
    data_path = args.root_path
    dataset_path = args.dataset_path
    loss_type = args.loss_type
    imbalance_ratio = args.imbalance_ratio
    linear_ratio = args.linear_ratio
    threshold = args.threshold
    device = torch.device(f'cuda:{args.gpuid}')

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logger = logging.getLogger(__name__)

    ranking_parameter = {
        'score_type': 'softmax',  # softmax: P(A), softmax_distance: P(A)-P(B), # new_distance
        'mean_score_type': 'mean',  # mean, taylor_mean
        're_correction': args.re_correction,
        # all_top, class_top, hybrid, class_hybrid, class_hybrid_LF, all_btm, class_btm
        're_sample_type': args.re_sample_type,
        'linear_ratio': linear_ratio,
        'score_threshold': threshold,
        're_sample_concat': True,
        'ndcg_ratio': 0.2,
        'dev_mode': False
    }
    #### Load dataset
    extract_fn = 'bert'
    model_name = 'roberta-base'
    # m_name = '_roberta'
    m_name = ''

    train_data, valid_data, test_data = load_dataset(dataset_path, data,
                                                     extract_feature=True,
                                                     extract_fn=extract_fn,
                                                     cache_name='roberta',
                                                     model_name=model_name)

    #### Create imbalanced dataset
    aggregated_hard_labels = np.load(f'{data_path}/label_model_output/pred_imbalance{imbalance_ratio}_{data}_hard.npy')

    train_ids = np.load(f'{data_path}/label_model_output/train_ids_imbalance{imbalance_ratio}_{data}.npy')
    valid_ids = np.load(f'{data_path}/label_model_output/valid_ids_imbalance{imbalance_ratio}_{data}.npy')
    train_data = train_data.create_subset(train_ids).get_covered_subset()
    valid_data = valid_data.create_subset(valid_ids)

    results = {}
    # for loss_type in ['normal']:
    #### Load optimized params
    with open(f"{data_path}/eval_res/optimized{m_name}_res_imbalance{imbalance_ratio}_{data}.json", "r") as load_f:
        optimized_param = json.load(load_f)

    #### Run end model: MLP
    print(optimized_param[loss_type])

    log = {}
    meter = AverageMeter(names=METRIC.keys())
    for i in range(5):
        model = EndClassifierModel(
            **default_parameter,
            **ranking_parameter,
            **optimized_param[loss_type]
        )
        with efficient_training(amp=True):
            history = model.fit(
                dataset_train=train_data,
                dataset_name=data,
                y_train=aggregated_hard_labels,
                dataset_valid=valid_data,
                evaluation_step=10,

                distillation=True,
                score_step=100,  # epoch
                avg_score_step=200,

                metric='f1_macro',
                patience=250,
                device=device
            )
        proba_y = model.predict_proba(test_data)
        metrics = {metric: metric_fn(test_data.labels, proba_y) for metric, metric_fn in METRIC.items()}
        logger.info(metrics)
        log[i] = {
            'metrics': metrics,
            'history': history,
        }
        meter.update(**metrics)

    metrics = meter.get_results()
    logger.info(metrics)
    best_log = {
        'metrics': metrics,
        'exp': log,
        'paras': optimized_param[loss_type]}
    results[loss_type] = best_log
    json.dump(results, open(f'{data_path}/eval_res/inference_distillation_mlp_{data}_{imbalance_ratio}_'
                            f'{ranking_parameter["score_type"]}.json', 'w'), indent=4)