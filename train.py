import os
import shutil

import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Additional Setup for MNIST-1D
# !git clone https://github.com/greydanus/mnist1d
import mnist1d
from mnist1d.data import get_templates, get_dataset_args, get_dataset
from mnist1d.utils import set_seed, plot_signals, ObjectView, from_pickle
from functions import run
import wandb

# Weights and biases
wandb.init(project="tail-lights-recognition-classification", entity="bramvanriessen")

# Weights and biases config
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 100
}

# Create save location
save_weights = 'weights'  # Model weights will be saved here.
shutil.rmtree(save_weights, ignore_errors=True)
os.makedirs(save_weights)

# Load dataset
args = get_dataset_args()
data = get_dataset(args, path='./mnist1d_data.pkl', download=True)  # This is the default setting

# Set the batch size for training & testing
b_size = 100

# Convert 1D MNIST data to pytorch tensors
tensors_train = torch.Tensor(data['x']), torch.Tensor(data['y']).long()
tensors_test = torch.Tensor(data['x_test']), torch.Tensor(data['y_test']).long()

# Create training set and test set from tensors
train_set = TensorDataset(*tensors_train)
test_set = TensorDataset(*tensors_test)

# Create dataloaders from the training and test set for easier iteration over the data
train_loader = DataLoader(train_set, batch_size=b_size)
test_loader = DataLoader(test_set, batch_size=b_size)

# Get some data and check for dimensions
data_input, label = next(iter(train_loader))

# Check whether the data has the right dimensions
# assert(input.shape == torch.Size([b_size, 40]))
# assert(label.shape == torch.Size([b_size]))

# Display samples from dataset
templates = get_templates()
fig = plot_signals(templates['x'], templates['t'], labels=templates['y'],
                   args=get_dataset_args(), ratio=2.2, do_transform=True)

run('lstm', train_loader, test_loader, save_weights)
