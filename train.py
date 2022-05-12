# Setup
import os
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Additional Setup to use Tensorboard
!pip install -q tensorflow
%load_ext tensorboard

#Additional Setup for MNIST-1D
# !git clone https://github.com/greydanus/mnist1d
import mnist1d
from mnist1d.data import get_templates, get_dataset_args, get_dataset
from mnist1d.utils import set_seed, plot_signals, ObjectView, from_pickle

from functions import run

# Load dataset
args = get_dataset_args()
data = get_dataset(args, path='./mnist1d_data.pkl', download=True) # This is the default setting

# Set the batch size for training & testing
b_size = 100

# Convert 1D MNIST data to pytorch tensors
tensors_train = torch.Tensor(data['x']), torch.Tensor(data['y']).long()
tensors_test = torch.Tensor(data['x_test']),torch.Tensor(data['y_test']).long()

# Create training set and test set from tensors
train_set = TensorDataset(*tensors_train)
test_set = TensorDataset(*tensors_test)

# Create dataloaders from the training and test set for easier iteration over the data
train_loader = DataLoader(train_set, batch_size=b_size)
test_loader = DataLoader(test_set, batch_size=b_size)

# Get some data and check for dimensions
input, label = next(iter(train_loader))

# Check whether the data has the right dimensions
# assert(input.shape == torch.Size([b_size, 40]))
# assert(label.shape == torch.Size([b_size]))

# Display samples from dataset
templates = get_templates()
fig = plot_signals(templates['x'], templates['t'], labels=templates['y'],
                   args=get_dataset_args(), ratio=2.2, do_transform=True)

run('lstm')