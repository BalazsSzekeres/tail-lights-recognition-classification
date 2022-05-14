import os
import shutil
import torch
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from functions import run_test
from net.Net import Net
from net.LSTM import LSTM

# Additional Setup for MNIST-1D
# !git clone https://github.com/greydanus/mnist1d
import mnist1d
from mnist1d.data import get_templates, get_dataset_args, get_dataset
from mnist1d.utils import set_seed, plot_signals, ObjectView, from_pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_images = 'test_images'
shutil.rmtree(save_images, ignore_errors=True)
os.makedirs(save_images)

# Load dataset
args = get_dataset_args()
data = get_dataset(args, path='./mnist1d_data.pkl', download=True)  # This is the default setting

# Set the batch size for training & testing
b_size = 100

# Convert 1D MNIST data to pytorch tensors
tensors_test = torch.Tensor(data['x_test']), torch.Tensor(data['y_test']).long()
test_set = TensorDataset(*tensors_test)

# Create dataloaders from the test set for easier iteration over the data
test_loader = DataLoader(test_set, batch_size=b_size)
rnn_type = 'lstm'

if rnn_type == 'lstm':
    hidden_size = 10
    rnn = LSTM(1, hidden_size)
    # Create classifier model
    model = nn.Sequential(OrderedDict([
        ('reshape', nn.Unflatten(1, (40, 1))),
        ('rnn', rnn),
        ('flat', nn.Flatten()),
        ('classifier', nn.Linear(40 * hidden_size, 10))
    ]))
else:
    config_file = "config/net.yaml"
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    model = Net(config)

print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
checkpoint = torch.load('weights/weights_100', map_location=device)
model.load_state_dict(checkpoint['model'])

run_test(model, test_loader, save_images)
print('Test finished')
