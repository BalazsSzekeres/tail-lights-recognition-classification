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

import wandb


class Runner:
    def __init__(self, train_loader, test_loader, config, device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.writer = SummaryWriter()
        net_config = yaml.safe_load(open(os.path.join("config", "net", config["net"])).read())
        self.model = Net(net_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"])

    def train(self):
        avg_loss = 0
        correct = 0
        total = 0

        # Iterate through batches
        for data in self.train_loader:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Move data to target device
            labels = labels.to(self.device)
            # Trying out some different things for input
            # inputs = [seq.to(self.device) for seq in inputs]
            # outputs = torch.empty(len(inputs), 2).to(self.device)
            outputs = torch.empty(len(inputs), 2).to(self.device)
            for j in range(len(inputs)):
                outputs[j] = self.model(inputs[j])

            # Forward + backward + optimize
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Keep track of loss and accuracy
            avg_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return avg_loss / len(self.train_loader), 100 * correct / total

    def test(self):
        avg_loss = 0
        correct = 0
        total = 0

        # Use torch.no_grad to skip gradient calculation, not needed for evaluation
        with torch.no_grad():
            # Iterate through batches
            for data in self.test_loader:
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Move data to target device
                labels = labels.to(self.device)
                # Trying out some different things for input
                # inputs = inputs.to(self.device)
                # inputs = [seq.to(self.device) for seq in inputs]
                outputs = torch.empty(len(inputs), 2).to(self.device)
                for i, input in enumerate(inputs):
                    outputs[i] = self.model(input)

                # Forward pass
                loss = self.criterion(outputs, labels)

                # Keep track of loss and accuracy
                avg_loss += float(loss)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return avg_loss / len(self.test_loader), 100 * correct / total

    def run(self):
        for epoch in tqdm(range(self.config["epochs"])):
            # Train on data
            train_loss, train_acc = self.train()
            # Test on data
            test_loss, test_acc = self.test()

            # print(f"Epoch {epoch}:\n"
            #       f"\t{train_loss=}, {train_acc=}"
            #       f"\t{test_loss=}, {test_acc=}")

            # # Write metrics to Tensorboard
            # writer.add_scalars('Loss', {
            #     'Train_{}'.format(model_type): train_loss,
            #     'Test_{}'.format(model_type): test_loss
            # }, epoch)
            #
            # writer.add_scalars('Accuracy', {
            #     'Train_{}'.format(model_type): train_acc,
            #     'Test_{}'.format(model_type): test_acc
            # }, epoch)

            # Write metrics to Weights and Biases
            model_type = self.config["net"][:-5]

            if self.config["wandb"]:
                wandb.log({'Train_Loss_{}'.format(model_type): train_loss})
                wandb.log({'Test_Loss_{}'.format(model_type): test_loss})
                wandb.log({'Train_Accuracy_{}'.format(model_type): train_acc})
                wandb.log({'Test_Accuracy_{}'.format(model_type): test_acc})

            # # Optional
            # wandb.watch(model)

            if self.config["save_weights"]:
                torch.save({'model': self.model.state_dict()}, os.path.join('weights', f'weights_{epoch}.pt'))
        print('\nFinished.')
        self.writer.flush()
        self.writer.close()

    def evaluate(self, save_csv_files):
        with torch.no_grad():
            self.model.eval()
            accuracy_list = []
            for index, img in enumerate(self.test_loader):
                images, labels = img
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(images)
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

            # np.savetxt(os.path.join(save_csv_files, 'Test_avg_accuracy_{}.csv'.format(accuracy_avg)),
            #            [p for p in zip(labels.cpu(), pred_indices.cpu(), (pred_indices == labels).cpu())],
            #            header='Labels, Prediction, Result', delimiter=',', fmt='%s')
