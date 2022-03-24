import argparse
import logging
import pickle

import torch
import json
import numpy as np
import sys
sys.path.append('../')
from wrench import efficient_training
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
# loss modified
from ars2 import EndClassifierModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='../')
    parser.add_argument("--dataset_path", type=str, default='../datasets/')
    parser.add_argument("--teacher_path", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--real_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--imbalance_ratio", type=int, default=10)
    parser.add_argument("--re_sample_type", type=str, default='class_top')
    parser.add_argument("--re_correction", type=bool, default=True)
    parser.add_argument("--linear_ratio", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--loss_type", type=str, default='normal_la')
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu")
    args = parser.parse_args()

    data = args.data
    data_path = args.root_path
    dataset_path = args.dataset_path
    loss_type = args.loss_type
    imbalance_ratio = args.imbalance_ratio
    linear_ratio = args.linear_ratio
    threshold = args.threshold
    device = torch.device(f'cuda:{args.gpu}')

    # BERT student: roberta-base
    student_default_parameter = {
        'batch_size'               : args.batch_size,
        'real_batch_size'          : args.real_batch_size,
        'test_batch_size'          : args.test_batch_size,
        'n_steps'                  : args.n_steps,
        'grad_norm'                : -1,
        'use_lr_scheduler'         : True,
        'binary_mode'              : False,

        'lr_scheduler'             : 'default',
        'optimizer'                : 'default',

        'backbone'                 : 'BERT',
        'backbone_model_name'      : 'roberta-base',
        'backbone_max_tokens'      : 512,
        'backbone_fine_tune_layers': 4
    }

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logger = logging.getLogger(__name__)

    # BERT
    student_ranking_parameter = {
        're_sample_type': args.re_sample_type,  # all_top, class_top, hybrid, class_hybrid, class_hybrid_LF, all_btm, class_btm
        're_correction': args.re_correction,
        'score_type': 'softmax_distance',  # softmax: P(A), softmax_distance: P(A)-P(B)
        'mean_score_type': 'mean',  # mean, taylor_mean
        're_sample_concat': True,
        'linear_ratio': 300,
        'score_threshold': 0,
        'ndcg_ratio': 0.2,
        'dev_mode': True
    }
    #### Load dataset
    extract_fn = 'bert'
    model_name = 'roberta-base'
    m_name='co_teach'
    teacher_m_name = ''
    student_m_name = '_roberta'

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

    # for loss_type in ['normal', 'normal_la', 'en', 'en_la', 'dice', 'ldam']:
    #### Load optimized params
    with open(f"{data_path}/eval_res/optimized{student_m_name}_res_imbalance{imbalance_ratio}_{data}.json", "r") as load_f:
        student_optimized_param = json.load(load_f)

    #### Run end model: MLP
    res = {
        'teacher_acc': [],
        'teacher_f1': [],
        'student_acc': [],
        'student_f1': []
    }
    for _ in range(5):
        teacher_model = EndClassifierModel()
        teacher_model.load(args.teacher_path)

        student_model = EndClassifierModel(
            **student_default_parameter,
            **student_optimized_param[loss_type],
            **student_ranking_parameter
        )
        with efficient_training(amp=True):
            student_model.fit(
                world_size=1,

                dataset_train=train_data,
                dataset_name=data,
                y_train=aggregated_hard_labels,
                dataset_valid=valid_data,
                evaluation_step=5,
                score_step=5,  # epoch
                avg_score_step=10,

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

    logger.info(f'end model ({m_name} {loss_type}) test acc(avg): {np.mean(res["acc"])}({np.std(res["acc"])}), '
                f'test f1_macro(avg): {np.mean(res["f1"])}({np.std(res["f1"])})')

    with open(f"{data_path}/eval_res/ts_distillation_{m_name}_res_imbalance{imbalance_ratio}_{data}.json", "a+") as f:
        f.write(f'Teacher ({m_name} {loss_type}) test acc(avg): {np.mean(res["teacher_acc"])}({np.std(res["teacher_acc"])}), '
                f'test f1_macro(avg): {np.mean(res["teacher_f1"])}({np.std(res["teacher_f1"])}) \n')
        f.write(
            f'Student ({m_name} {loss_type}) test acc(avg): {np.mean(res["student_acc"])}({np.std(res["student_acc"])}), '
            f'test f1_macro(avg): {np.mean(res["student_f1"])}({np.std(res["student_f1"])}) \n')
