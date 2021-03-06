import os, csv
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
import pdb
from collections import defaultdict
from datetime import datetime

# wilds github
sys.path.append("/home/johnny/wilds/examples")
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, \
        initialize_wandb, log_group_data, parse_bool, get_model_prefix
from train import train, evaluate
from algorithms.initializer import initialize_algorithm
from transforms import initialize_transform
from configs.utils import populate_defaults
import configs.supported as supported

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to downloads the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str)

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
        help='keyword arguments for model initialization passed as key1=value1 key2=value2')

    # Transforms
    parser.add_argument('--train_transform', choices=supported.transforms)
    parser.add_argument('--eval_transform', choices=supported.transforms)
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)

    # Objective
    parser.add_argument('--loss_function', choices = supported.losses)

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--algo_log_metric')

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={})

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False)

    # active learning
    parser.add_argument('--total_data', type=int, required=True)
    parser.add_argument('--valid_size', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--start_balanced', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--al_strategy', type=str, required=True, choices = ['rs','us', 'cs'])
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--patience', type=int, required=True)

    config = parser.parse_args()
    config = populate_defaults(config)

    # set device
    config.device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    ## Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    if config.seed:
        set_seed(config.seed)
    else:
        date = datetime.now()
        set_seed(date.microsecond)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    # To implement data augmentation (i.e., have different transforms
    # at training time vs. test time), modify these two lines:
    train_transform = initialize_transform(
        transform_name=config.train_transform,
        config=config,
        dataset=full_dataset)
    eval_transform = initialize_transform(
        transform_name=config.eval_transform,
        config=config,
        dataset=full_dataset)

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields)

    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset

        if split =='val':
            num_ex = len(full_dataset.get_subset('val'))
            frac = config.valid_size / num_ex
        else:
            frac = config.frac

        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=frac,
            transform=transform)
        print(frac)
        print('%s: %d examples.' % (split, len(datasets[split]['dataset'])))

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))

        if config.use_wandb:
            initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)


    # do active learning here
    from active_learning_wilds import active_learning, \
            random_sampling, uncertainty_sampling, \
            certainty_sampling

    classes = np.unique(datasets['train']['dataset'].y_array)
    strategies = {'rs':random_sampling, 'us':uncertainty_sampling, 'cs':certainty_sampling}

    al_strategy = strategies[config.al_strategy]
    results = active_learning(al_strategy, full_dataset, datasets, np.arange(len(datasets['train']['dataset'])), config, train_grouper,
            classes=classes)

    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

    results.save(os.path.join(config.log_dir, '%s_%s_%s.jsonl' % (config.dataset, config.al_strategy, config.exp_id)))

if __name__=='__main__':
    main()
