import numpy as np
from tqdm import tqdm
import scipy
import copy
import json
import pdb
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from algorithms.initializer import initialize_algorithm
from train import train, evaluate

def test(model, test_loader, device):
  test_loss = 0
  correct = 0
  model.eval()
  with torch.no_grad():
    predictions = []
    for batch in tqdm(test_loader):
      data, target, groups = batch[0], batch[1], batch[2]
      data = data.to(device)
      output = model.evaluate(batch)['y_pred']
      test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      predictions.append(output.cpu())
  predictions = np.concatenate(predictions)
  acc = correct / len(predictions)
  return test_loss, acc, predictions

def sample_balanced(pool_idx, n_per_class, targets, classes):
  idxs = []
  for c in classes:
    pool_class_idx = pool_idx[targets == c]
    idx = np.random.choice(pool_class_idx, size=n_per_class)
    idxs.append(idx)
  return np.concatenate(idxs)

def random_sampling(model, train_loader, pool_idx, n, device):
  return np.random.choice(pool_idx, size=n)

def uncertainty_sampling(model, train_loader, pool_idx, n, device):
  # generate answers
  test_loss, acc, outputs = test(model, train_loader, device)

  # max entropy
  H = scipy.stats.entropy(scipy.special.softmax(outputs, axis=1), axis=1)
  sorted_idx = H.argsort()
  sorted_idx = sorted_idx[np.isin(sorted_idx, pool_idx)]

  # larger is more uncertain
  return sorted_idx[:-n-1:-1]


def active_learning(sampling_func, full_dataset, datasets, pool_idx, 
        config, train_grouper, classes=[0,1], verbose=True, test_measure='acc_wg', **kwargs):
  # sample balanced
  DATA_PER_ROUND = int(config.total_data / config.rounds)
  train_idx = sample_balanced(pool_idx, 
                              int(DATA_PER_ROUND/len(classes)), 
                              datasets['train']['dataset'].y_array[pool_idx], 
                              classes)
  pool_idx = np.setdiff1d(pool_idx, train_idx)
  
  results = []

  for round in range(1, config.rounds+1):
    print('Round %d: %d training points' % (round, len(train_idx)))
    train_sampler = SubsetRandomSampler(train_idx)
    it_train_loader = DataLoader(
        datasets['train']['dataset'],
        shuffle=False, # Shuffle training dataset
        sampler=train_sampler,
        collate_fn=datasets['train']['dataset'].collate,
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
    al_datasets['val']['loader'] = [next(iter(datasets['val']['loader']))]

    train(
        algorithm=algorithm,
        datasets=al_datasets,
        general_logger=open('log.txt', 'a'),
        config=config,
        epoch_offset=0,
        best_val_metric=None)

    # test
    test_loss, test_acc, predictions = test(algorithm, datasets['test']['loader'], config.device)
    acc_wg = full_dataset.eval(
        torch.Tensor(predictions.argmax(axis=1)), 
        datasets['test']['dataset'].y_array,
        datasets['test']['dataset'].metadata_array,
      )[0][test_measure]

    log_file = 'results_%s_%d' % (config.exp_id, len(train_idx))
    with open(path.join(config.log_dir, log_file), 'wt') as fh:
        fh.write('test_acc\t%f\n' % test_acc)
        fh.write('%s\t%f\n' % (test_measure, acc_wg))
        fh.write('train_idx\t%s\n' % json.dumps(train_idx.tolist()))
    
    # sample
    new_idx = sampling_func(algorithm, datasets['train']['loader'], pool_idx, DATA_PER_ROUND, config.device)
    train_idx = np.concatenate([train_idx, new_idx])
    pool_idx = np.setdiff1d(pool_idx, train_idx)

    if verbose:
      print('Test loss %.2f, test acc %.2f' % (test_loss, test_acc))
      print()