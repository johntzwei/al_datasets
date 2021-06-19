import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import scipy
import pdb

def test(model, test_loader, device):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        predictions = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()    # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.append(pred.cpu())
    predictions = np.concatenate(predictions).reshape(-1)
    acc = correct / len(predictions)
    return test_loss, acc, predictions

def train(model, train_loader, valid_loader, device, 
                    EPOCHS=50, LOG_INTERVAL=10, PATIENCE=5, verbose=True):
    optimizer = optim.Adam(model.parameters(), weight_decay=3/len(train_loader.dataset))
    batch_idx = 0
    best_acc, patience = -1, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            batch_idx += 1
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # calculate validation
        valid_loss, valid_acc, predictions = test(model, valid_loader, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
        else:
            patience += 1

        if epoch % LOG_INTERVAL == 1 and epoch != 1 and verbose:
            print(' | Epoch %d: valid loss %.2f, valid accuracy %.2f' % (epoch, valid_loss, valid_acc))

        if patience == PATIENCE:
            break
    
    if verbose:
        print(' | Final epoch %d: valid loss %.2f, valid accuracy %.2f' % (epoch, valid_loss, valid_acc))

def random_sampling(model, train_loader, pool_idx, n, device):
    return np.random.choice(pool_idx, size=n)

def uncertainty_sampling(model, train_loader, pool_idx, n, device):
    # generate answers
    outputs = []
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output.cpu())
    
    outputs = np.concatenate(outputs)

    # max entropy
    H = scipy.stats.entropy(scipy.special.softmax(outputs, axis=1), axis=1)
    sorted_idx = H.argsort()
    sorted_idx = sorted_idx[np.isin(sorted_idx, pool_idx)]

    # larger is more uncertain
    return sorted_idx[:-n-1:-1]

def sample_balanced(pool_idx, n_per_class, targets, classes):
    idxs = []
    for c in classes:
        pool_class_idx = pool_idx[targets == c]
        idx = np.random.choice(pool_class_idx, size=n_per_class)
        idxs.append(idx)
    return np.concatenate(idxs)

def active_learning(sampling_func, model_class, dataset1, train_loader, valid_loader, test_loader, classes, pool_idx, device, verbose=True, ROUNDS = 50, TOTAL_DATA = 1000, **kwargs):
    # sample balanced
    DATA_PER_ROUND = int(TOTAL_DATA / ROUNDS)
    train_idx = sample_balanced(pool_idx, int(DATA_PER_ROUND/len(classes)), dataset1.targets[pool_idx], classes)
    pool_idx = np.setdiff1d(pool_idx, train_idx)
    
    results = []

    for round in range(1, ROUNDS+1):
        print('Round %d: %d training points' % (round, len(train_idx)))
        train_sampler = SubsetRandomSampler(train_idx)
        it_train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, batch_size=int(len(train_idx)/20))

        # train
        model = model_class()
        model.to(device)
        train(model, it_train_loader, valid_loader, device, **kwargs)

        # test
        test_loss, test_acc, predictions = test(model, test_loader, device)
        results.append((len(train_idx), test_acc))
        
        # sample
        new_idx = sampling_func(model, train_loader, pool_idx, DATA_PER_ROUND, device)
        train_idx = np.concatenate([train_idx, new_idx])
        pool_idx = np.setdiff1d(pool_idx, train_idx)

        if verbose:
            print('Test loss %.2f, test acc %.2f' % (test_loss, test_acc))
            print()

    return results
