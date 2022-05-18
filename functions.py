import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
# Additional Setup to use Tensorboard
# !pip install -q tensorflow
# %load_ext tensorboard
from torch.utils.tensorboard import SummaryWriter
from net.Net import Net
from net.LSTM import LSTM
import yaml
import numpy as np


# import wandb


def train(train_loader, model, optimizer, criterion, device):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
        device: Running device (e.g. cuda or cpu)
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for i, data in enumerate(train_loader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move data to target device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


def test(test_loader, model, criterion, device):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for data in test_loader:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Move data to target device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total


def run(model_type, trainloader, testloader, weights_location, epochs=100):
    """
    Run a test on MNIST-1D

    Args:
        weights_location: location for saving weights
        trainloader: training data
        testloader: test data
        model_type: type of model
        epochs: number of epochs to run
    """
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    model = config_model(model_type)

    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 1e-2, weight_decay=1e-3)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        # Train on data
        train_loss, train_acc = train(trainloader,
                                      model,
                                      optimizer,
                                      criterion,
                                      device)

        # Test on data
        test_loss, test_acc = test(testloader,
                                   model,
                                   criterion,
                                   device)

        # Write metrics to Tensorboard
        writer.add_scalars('Loss', {
            'Train_{}'.format(model_type): train_loss,
            'Test_{}'.format(model_type): test_loss
        }, epoch)

        writer.add_scalars('Accuracy', {
            'Train_{}'.format(model_type): train_acc,
            'Test_{}'.format(model_type): test_acc
        }, epoch)

        # Write metrics to Weights and Biases
        # wandb.log({'Train_Loss_{}'.format(model_type): train_loss})
        # wandb.log({'Test_Loss_{}'.format(model_type): test_loss})
        # wandb.log({'Train_Accuracy_{}'.format(model_type): train_acc})
        # wandb.log({'Test_Accuracy_{}'.format(model_type): test_acc})

        # # Optional
        # wandb.watch(model)

    torch.save({'model': model.state_dict()}, os.path.join(weights_location, 'weights_{}'.format(epochs)))
    print('\nFinished.')
    writer.flush()
    writer.close()


def evaluate(model, dataloader_test, save_csv_files, device):
    """
    Run a test on MNIST-1D

    Args:
        save_csv_files: CSV file location
        device: Device is either cpu or cuda
        dataloader_test: Test set
        model: Model to evaluate
    """
    with torch.no_grad():
        model.eval()
        accuracy_list = []
        for index, img in enumerate(dataloader_test):
            images, labels = img
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            pred_indices = torch.argmax(pred, 1)
            accuracy = (pred_indices == labels).sum().item() / labels.size(0)
            # print('Predictions: ', pred_indices)
            # print('Labels: ', labels)
            # print('Results: ', pred_indices == labels)
            # print('Accuracy: ', accuracy)
            accuracy_list.append(accuracy)
            # print('Predictions size: ', pred.size())
            # print("label: {}, pred: {}".format(label, pred))

        accuracy_avg = sum(accuracy_list) / len(accuracy_list)
        print('Average accuracy: ', accuracy_avg)

        np.savetxt(os.path.join(save_csv_files, 'Test_avg_accuracy_{}.csv'.format(accuracy_avg)),
                   [p for p in zip(labels.cpu(), pred_indices.cpu(), (pred_indices == labels).cpu())],
                   header='Labels, Prediction, Result', delimiter=',', fmt='%s')


def config_model(model_type, hidden_size=10):
    """
    Configure model based on given model type

    Args:
        model_type: Type of model (eg. LSTM)
        hidden_size: Hidden size or layer

    Returns:
        model
    """
    if model_type == 'lstm':
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
    return model
