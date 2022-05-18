import os
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader
from functions import evaluate, config_model

# Additional Setup for MNIST-1D
# !git clone https://github.com/greydanus/mnist1d
from mnist1d.data import get_dataset_args, get_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_images = 'test_images'
save_csv_files = ''
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
model_type = 'lstm'

model = config_model(model_type)

print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
checkpoint = torch.load('weights/weights_100', map_location=device)
model.load_state_dict(checkpoint['model'])

evaluate(model, test_loader, save_csv_files, device)
print('Test finished')
