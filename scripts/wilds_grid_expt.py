import os, csv
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
import copy
import pdb
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# wilds github
sys.path.append("/home/johnny/wilds/examples")
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from active_learning_wilds import al_results, test

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
    parser.add_argument('--seed', type=int, default=0)
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

    parser.add_argument('--valid_size', type=int, default=None)
    parser.add_argument('--minority', type=str, default=None)
    parser.add_argument('--majority', type=str, default=None)
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--patience', type=int, required=True)

    parser.add_argument('--grid_index', type=int, required=True)
    parser.add_argument('--ex_step_size', type=int, required=True)
    parser.add_argument('--num_grid_rc', type=int, required=True)

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
    set_seed(hash(config.exp_id) % 1000000)

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
            frac = 1
        else:
            frac = config.frac

        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=frac,
            transform=transform)
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


    # get grid index
    idx = config.grid_index % (config.num_grid_rc ** 2)
    row, col = int(idx / config.num_grid_rc), idx % config.num_grid_rc
    min_ex, maj_ex = row * config.ex_step_size, col * config.ex_step_size


    # get the training data here
    def get_idx(dataset, minority='black', majority='white', min_ex=0, maj_ex=0, valid=False):
        metadata_array = dataset.metadata_array
        min_group = dataset.metadata_fields.index(minority)
        maj_group = dataset.metadata_fields.index(majority)

        if minority != majority:
            minority_idx = np.arange(0, len(metadata_array))[(metadata_array[:,min_group] == 1) & (metadata_array[:,maj_group] == 0)]
            majority_idx = np.arange(0, len(metadata_array))[(metadata_array[:,min_group] == 0) & (metadata_array[:,maj_group] == 1)]
        else:
            minority_idx = np.arange(0, len(metadata_array))[(metadata_array[:,min_group] == 1)]
            majority_idx = np.arange(0, len(metadata_array))[(metadata_array[:,min_group] == 0)]

        
        if valid:
            train_idx = minority_idx

            # for mnli topics
            size = min(len(minority_idx), len(majority_idx))

            train_idx = np.concatenate([train_idx, \
                    np.random.choice(majority_idx, size=size, replace=False)])
            return train_idx
        else:
            # train
            train_idx = np.concatenate([ \
                    np.random.choice(minority_idx, size=min_ex, replace=False), \
                    np.random.choice(majority_idx, size=maj_ex, replace=False)])
            return train_idx

    train_dataset = datasets['train']['dataset']
    train_idx = get_idx(train_dataset, minority=config.minority, majority=config.majority, min_ex=min_ex, maj_ex=maj_ex)
    print('Training data: %d' % len(train_idx))
    print('Minority examples: %d, majority examples: %d' % (min_ex, maj_ex))

    valid_dataset = datasets['val']['dataset']
    valid_idx = get_idx(valid_dataset, minority=config.minority, majority=config.majority, valid=True)
    print('Valid data: %d' % len(valid_idx))


    # do subsampling here
    train_sampler = SubsetRandomSampler(train_idx)
    it_train_loader = DataLoader(
            datasets['train']['dataset'],
            shuffle=False, # Shuffle training dataset
            sampler=train_sampler,
            collate_fn=datasets['train']['dataset'].collate,
            batch_size=config.batch_size)

    valid_sampler = SubsetRandomSampler(valid_idx)
    it_valid_loader = DataLoader(
            datasets['val']['dataset'],
            shuffle=False, # Shuffle training dataset
            sampler=valid_sampler,
            collate_fn=datasets['val']['dataset'].collate,
            batch_size=config.batch_size)



    # train
    algorithm = initialize_algorithm(
            config=config,
            datasets=datasets,
            train_grouper=train_grouper)
    
    al_datasets = copy.copy(datasets)
    al_datasets['train'] = copy.copy(datasets['train'])
    al_datasets['val'] = copy.copy(datasets['val'])
    al_datasets['train']['loader'] = it_train_loader
    al_datasets['val']['loader'] = it_valid_loader

    train(
            algorithm=algorithm,
            datasets=al_datasets,
            general_logger=open('log.txt', 'a'),
            config=config,
            epoch_offset=0,
            best_val_metric=None)

    # test
    test_loss, test_acc, predictions = test(algorithm, datasets['test']['loader'], config.device)
    evals = full_dataset.eval(
            torch.Tensor(predictions.argmax(axis=1)), 
            datasets['test']['dataset'].y_array,
            datasets['test']['dataset'].metadata_array,
        )[0]


    # write results
    results = al_results()
    results.add_round()
    results.add_result('train_size', len(train_idx))
    results.add_result('train_idx', train_idx.tolist())
    results.add_result('valid_idx', valid_idx.tolist())
    results.add_result('test_acc', test_acc)
    results.add_result('test_outs', predictions.tolist())
    results.add_result('evals', evals)

    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

    results.save(os.path.join(config.log_dir, '%s_%d_%d_%s_%s_%s.jsonl' % (config.dataset, min_ex, maj_ex, config.exp_id, config.minority, config.majority)))

if __name__=='__main__':
    main()
