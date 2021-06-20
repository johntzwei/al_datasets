import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import os, sys
sys.path.append("/home/johnny/al_datasets/scripts")
from active_learning import train, test, \
        random_sampling, uncertainty_sampling, \
        sample_balanced, active_learning

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import scipy
import pdb

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('-e', '--experiment_id', type=int, required=True)
parser.add_argument('-p', '--experiment_path', type=str, required=True)
parser.add_argument('-s', '--al_sampling_method', type=str, required=True)
parser.add_argument('--valid_size', type=int, default=100)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--total_data', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size)

dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device('cuda')
print(torch.cuda.get_device_name())

# create valid and pool
valid_idx = np.random.choice(np.arange(0, len(dataset1)), args.valid_size)
valid_sampler = SubsetRandomSampler(valid_idx)
valid_loader = torch.utils.data.DataLoader(dataset1, sampler=valid_sampler, batch_size=args.batch_size)
pool_idx = np.setdiff1d(np.arange(0, len(dataset1)), valid_idx)

if args.al_sampling_method == 'random':
    sampling_method = random_sampling
elif args.al_sampling_method == 'uncertainty':
    sampling_method = uncertainty_sampling

results = active_learning(sampling_method, Net, dataset1, train_loader, valid_loader, test_loader, classes, pool_idx, device, ROUNDS=args.rounds, TOTAL_DATA=args.total_data, EPOCHS=args.epochs, PATIENCE=args.patience, batch_size=args.batch_size)
results.save(os.path.join(args.experiment_path, 'mnist_%s_%d.jsonl' % (args.al_sampling_method, args.experiment_id)))
