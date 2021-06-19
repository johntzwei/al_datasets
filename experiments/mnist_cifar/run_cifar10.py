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
sys.path.append("/home/johnny/al_datasets/experiments")
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

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=128)

dataset2 = datasets.CIFAR10('../data', train=True,
                        transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=128)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

device = torch.device('cuda:3')
print(torch.cuda.get_device_name())

# create valid and pool
valid_idx = np.random.choice(np.arange(0, len(dataset1)), 1000)
valid_sampler = SubsetRandomSampler(valid_idx)
valid_loader = torch.utils.data.DataLoader(dataset1, sampler=valid_sampler, batch_size=128)
pool_idx = np.setdiff1d(np.arange(0, len(dataset1)), valid_idx)

dataset1.targets = np.array(dataset1.targets)
results = active_learning(random_sampling, Net, dataset1, valid_loader, train_loader, test_loader, classes, pool_idx, device, TOTAL_DATA=200, ROUNDS=5, EPOCHS=100, PATIENCE=50)
