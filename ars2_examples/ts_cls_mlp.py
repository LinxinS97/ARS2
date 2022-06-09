import logging
import argparse
import torch
import json
import numpy as np
import sys


# parameters of MLP
default_parameter = {
    'batch_size'            : 64,  # TREC: 64, Chemprot: 64
    'real_batch_size'       : 64,
    'test_batch_size'       : 1024,
    'warm_up_steps'         : 1500,  # TREC: 5000, Chemprot: 1500
    'n_steps'               : 100000,
    'use_lr_scheduler'      : True,
    'lr_scheduler'          : 'default',
    'optimizer'             : 'default',
    'binary_mode'           : False,
    'backbone'              : 'MLP',
    'backbone_n_layers'     : 2,
    'backbone_hidden_size'  : 128
}

# optimal hyperparameter of ARS2
hyper_params = {
    'agnews': {
        '1': {
            'linear_ratio': 3,
            'score_threshold': 0.3,
        },
        '10': {
            'linear_ratio': 1,
            'score_threshold': 0,
        },
        '20': {
            'linear_ratio': 0.9,
            'score_threshold': 0,
        },
        '50': {
            'linear_ratio': 1,
            'score_threshold': 0,
        }
    },
    'yelp': {
        '1': {
            'linear_ratio': 1,
            'score_threshold': 0,
        },
        '10': {
            'linear_ratio': 0.9,
            'score_threshold': -0.1,
        },
        '20': {
            'linear_ratio': 0.9,
            'score_threshold': -0.1,
        },
        '50': {
            'linear_ratio': 0.9,
            'score_threshold': -0.3,
        }
    },
    'trec': {
        '1': {
            'linear_ratio': 3,
            'score_threshold': -1,
        }
    },
    'chemprot': {
        '1': {
            'linear_ratio': 1,
            'score_threshold': 0.2,
        }
    }
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_root', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--lm_pred_path', type=str, default='./', help="path of the prediction of label model.")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--optimal_params_path', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--ir', type=str, help='imbalance ratio', required=True)
    parser.add_argument('--ars2', type=bool, default=False, help='selection for opening ars2')
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu (e.g. 0)')
    args = parser.parse_args()

    # path to root
    if args.path_to_root:
        ptr = args.path_to_root
    else:
        ptr = './'

    from wrench.evaluation import AverageMeter, METRIC
    from wrench.dataset import load_dataset
    from wrench.logging import LoggingHandler

    # loss modified
    from ars2 import EndClassifierModel

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logger = logging.getLogger(__name__)

    sys.path.append(args.path_to_root)
    dataset_path = f'{args.path_to_root}/{args.dataset_path}'
    lm_path = f'{args.path_to_root}/{args.lm_pred_path}'
    optimal_param_path = f'{args.path_to_root}/{args.optimal_params_path}'
    data = args.data
    imbalance_ratio = args.ir
    loss_type = args.loss_type
    ars2 = args.ars2

    device = torch.device(f'cuda:{args.gpu}')

    if ars2:
        # (RR, CR) For ablation: w/o CR&RR, w/o RR, w/o CR
        # cr_rr = [(False, 'all_top'), (False, 'class_top'), (True, None)]
        cr_rr = [(True, 'class_top')]  # (RR, CR)
    else:
        cr_rr = [(False, None)]  # (RR, CR)

    for para in cr_rr:
        ranking_parameter = {
            'score_type': 'pred',  # pred: P(A), margin(AUM): P(A)-P(B)
            'mean_score_type': 'mean',
            're_sample_type': para[1],  # switch of class-wise ranking
            're_sample_concat': False,  # Chemprot: False
            're_correction': para[0],  # switch of rule-aware ranking
            'ndcg_ratio': 0.2,
            'dev_mode': True
        }
        ranking_parameter.update(hyper_params[data][imbalance_ratio])

        #### Load dataset
        extract_fn = 'bert'
        model_name = 'roberta-base'
        m_name = ''

        train_data, valid_data, test_data = load_dataset(dataset_path, data,
                                                         extract_feature=True,
                                                         extract_fn=extract_fn,
                                                         cache_name='roberta',
                                                         model_name=model_name)

        #### Create imbalanced dataset
        aggregated_hard_labels = np.load(f'{lm_path}/pred_imbalance{imbalance_ratio}_{data}_hard.npy')

        train_ids = np.load(f'{lm_path}/train_ids_imbalance{imbalance_ratio}_{data}.npy')
        valid_ids = np.load(f'{lm_path}/valid_ids_imbalance{imbalance_ratio}_{data}.npy')
        train_data = train_data.create_subset(train_ids).get_covered_subset()
        valid_data = valid_data.create_subset(valid_ids)

        #### Load optimized params
        with open(f"{optimal_param_path}/optimized{m_name}_res_imbalance{imbalance_ratio}_{data}.json", "r") as load_f:
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
            history = model.fit(
                dataset_train=train_data,
                dataset_name=data,
                y_train=aggregated_hard_labels,
                dataset_valid=valid_data,
                evaluation_step=10,

                distillation=ars2,
                score_step=100,  # epoch
                avg_score_step=1000,  # 1000 for chemprot

                metric='f1_macro',
                patience=600,  # TREC: 600, Chemprot: 300
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

        cr_rr_opt = '_'.join([str(x) for x in para])
        json.dump(best_log, open(f'{args.path_to_root}/eval_res/inference_distillation_mlp_'
                                 f'{data}_{imbalance_ratio}_{loss_type}_{cr_rr_opt}.json', 'w'), indent=4)
