import logging

import argparse
import torch
import json
import numpy as np
import sys

# optimal hyperparameter of ARS2
teacher_hyper_params = {
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
            'linear_ratio': 1,
            'score_threshold': -1,
        }
    },
    'chemprot': {
        '1': {
            'linear_ratio': 3,
            'score_threshold': -0.04,
        }
    }
}

student_hyper_params = {
    'agnews': {
        '1': {
            'linear_ratio': 300,
            'score_threshold': 0.3,
        },
        '10': {
            'linear_ratio': 200,
            'score_threshold': 0,
        },
        '20': {
            'linear_ratio': 300,
            'score_threshold': 0,
        },
        '50': {
            'linear_ratio': 2000,
            'score_threshold': 0,
        }
    },
    'yelp': {
        '1': {
            'linear_ratio': 300,
            'score_threshold': 0,
        },
        '10': {
            'linear_ratio': 500,
            'score_threshold': -0.1,
        },
        '20': {
            'linear_ratio': 500,
            'score_threshold': -0.1,
        },
        '50': {
            'linear_ratio': 300,
            'score_threshold': -0.3,
        }
    },
    'trec': {
        '1': {
            'linear_ratio': 300,
            'score_threshold': -1,
        }
    },
    'chemprot': {
        '1': {
            'linear_ratio': 500,
            'score_threshold': -0.04,
        }
    }
}

# BERT student
student_default_parameter = {
    'batch_size': 64,
    'real_batch_size': 64,
    'test_batch_size': 256,
    'n_steps': 10000,
    'grad_norm': -1,
    'use_lr_scheduler': True,
    'binary_mode': False,

    'lr_scheduler': 'default',
    'optimizer': 'default',

    'backbone': 'BERT',
    'backbone_model_name': 'roberta-base',
    'backbone_max_tokens': 512,
    'backbone_fine_tune_layers': 4
}

# MLP
teacher_default_parameter = {
    'batch_size': 64,  # Chemprot: 64
    'real_batch_size': 64,
    'test_batch_size': 1024,
    'n_steps': 100000,
    'warm_up_steps': 2000,  # TREC: 5000, Chemprot: 2000
    'use_lr_scheduler': True,
    'lr_scheduler': 'default',
    'optimizer': 'default',
    'binary_mode': False,
    'backbone': 'MLP',
    'backbone_n_layers': 2,
    'backbone_hidden_size': 128
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
    parser.add_argument('--gpu', type=int, default=0, help='id of gpu (e.g. 0)')
    args = parser.parse_args()

    # path to root
    if args.path_to_root:
        ptr = args.path_to_root
    else:
        ptr = './'

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

    device = torch.device(f'cuda:{args.gpu}')

    teacher_ranking_parameter = {
        'score_type': 'margin',  # pred: P(A), margin: P(A)-P(B)
        'mean_score_type': 'mean',  # mean, taylor_mean
        're_sample_type': 'class_top',
        # all_top, class_top, hybrid, class_hybrid, class_hybrid_LF, all_btm, class_btm
        're_sample_concat': False,  # chemprot Falsew
        're_correction': False,  # chemprot: False
        'ndcg_ratio': 0.2,
        'dev_mode': True
    }
    student_ranking_parameter = {
        'score_type': 'margin',  # pred: P(A), margin: P(A)-P(B)
        'mean_score_type': 'mean',  # mean, taylor_mean
        're_sample_type': 'class_top',  # all_top, class_top, hybrid, class_hybrid, class_hybrid_LF, all_btm, class_btm
        're_sample_concat': False,
        're_correction': False,  # Chemprot: False
        'ndcg_ratio': 0.2,
        'dev_mode': True
    }

    teacher_ranking_parameter.update(teacher_hyper_params[data][imbalance_ratio])
    student_ranking_parameter.update(student_hyper_params[data][imbalance_ratio])

    #### Load dataset
    extract_fn = 'bert'
    model_name = 'roberta-base'
    m_name = 'co_teach'
    teacher_m_name = ''
    student_m_name = '_roberta'

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
    with open(f"{optimal_param_path}/optimized{teacher_m_name}_res_imbalance{imbalance_ratio}_{data}.json",
              "r") as load_f:
        teacher_optimized_param = json.load(load_f)
    with open(f"{optimal_param_path}/optimized{student_m_name}_res_imbalance{imbalance_ratio}_{data}.json",
              "r") as load_f:
        student_optimized_param = json.load(load_f)

    #### Run end model: MLP
    res = {
        'teacher_acc': [],
        'teacher_f1': [],
        'student_acc': [],
        'student_f1': []
    }
    for _ in range(5):
        teacher_model = EndClassifierModel(
            **teacher_default_parameter,
            **teacher_optimized_param[loss_type],
            **teacher_ranking_parameter
        )

        teacher_model.fit(
            dataset_train=train_data,
            dataset_name=data,
            y_train=aggregated_hard_labels,
            dataset_valid=valid_data,
            evaluation_step=10,

            distillation=True,
            score_step=100,  # epoch
            avg_score_step=1000,  # chemprot: 1000

            metric='f1_macro',
            patience=300,
            device=device
        )
        acc = teacher_model.test(test_data, 'acc')
        res['teacher_acc'].append(acc)
        f1_ma = teacher_model.test(test_data, 'f1_macro')
        res['teacher_f1'].append(f1_ma)
        logger.info(f'end model (Teacher) test acc: {acc}, test f1_macro: {f1_ma}')

        student_model = EndClassifierModel(
            **student_default_parameter,
            **student_optimized_param[loss_type],
            **student_ranking_parameter
        )
        student_model.fit(
            world_size=1,

            dataset_train=train_data,
            dataset_name=data,
            y_train=aggregated_hard_labels,
            dataset_valid=valid_data,
            evaluation_step=5,
            score_step=5,  # epoch
            avg_score_step=20,  # chemprot: 20

            distillation=True,
            teacher_model=teacher_model,

            metric='f1_macro',
            patience=40,
            device=device
        )
        acc = student_model.test(test_data, 'acc')
        res['student_acc'].append(acc)
        f1_ma = student_model.test(test_data, 'f1_macro')
        res['student_f1'].append(f1_ma)
        logger.info(f'end model (Student) test acc: {acc}, test f1_macro: {f1_ma}')

    logger.info(f'end model ({m_name} {loss_type}) test f1_macro(avg): {np.mean(res["student_f1"])}({np.std(res["student_f1"])})')

    with open(f"{args.path_to_root}/eval_res/ts_distillation_{m_name}_res_"
              f"{loss_type}_imbalance{imbalance_ratio}_{data}.json", "a+") as f:

        f.write(f'Teacher ({m_name}) test acc(avg): '
                f'{np.mean(res["teacher_acc"])}({np.std(res["teacher_acc"])}), '
                f'test f1_macro(avg): {np.mean(res["teacher_f1"])}({np.std(res["teacher_f1"])}) \n')

        f.write(f'Student ({m_name} {loss_type}) test acc(avg): '
                f'{np.mean(res["student_acc"])}({np.std(res["student_acc"])}), '
                f'test f1_macro(avg): {np.mean(res["student_f1"])}({np.std(res["student_f1"])}) \n')
