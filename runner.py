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
    def __init__(self, train_loader, val_loader, test_loader, config, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.writer = SummaryWriter()
        net_config = yaml.safe_load(open(os.path.join("config", "net", config["net"])).read())
        self.model = Net(net_config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()
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

    def test(self, val):
        avg_loss = 0
        correct = 0
        total = 0

        # Select validation set or test set
        data_loader = self.val_loader if val else self.test_loader

        # Use torch.no_grad to skip gradient calculation, not needed for evaluation
        with torch.no_grad():
            # Iterate through batches
            for data in data_loader:
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Move data to target device
                labels = labels.to(self.device)

                outputs = torch.empty(len(inputs), 2).to(self.device)
                for j in range(len(inputs)):
                    outputs[j] = self.model(inputs[j])

                # Forward pass
                loss = self.criterion(outputs, labels)

                # Keep track of loss and accuracy
                avg_loss += float(loss)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return avg_loss / len(data_loader), 100 * correct / total

    def run(self):
        # Patience - how many epochs to keep training after accuracy has not improved
        patience = self.config["patience"]

        # Initialize early stopping variables
        test_acc_best = 0
        patience_cnt = 0

        for epoch in tqdm(range(self.config["epochs"])):
            # Train on data
            train_loss, train_acc = self.train()
            # Val on data
            val_loss, val_acc = self.test(val=True)

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

            log_dict = {'Train_Loss_{}'.format(model_type): train_loss,
                        'Val_Loss_{}'.format(model_type): val_loss,
                        'Train_Accuracy_{}'.format(model_type): train_acc,
                        'Val_Accuracy_{}'.format(model_type): val_acc}
            print(log_dict)
            if self.config["wandb"]:
                wandb.log(log_dict)

            # # Optional
            # wandb.watch(model)

            if self.config["save_weights"]:
                torch.save({'model': self.model.state_dict()}, os.path.join('weights', f'weights_{epoch}.pt'))

            if self.config["early_stopping"] == True:
                if val_acc > val_acc_best:
                    val_acc_best = val_acc
                    patience_cnt = 0
                else:
                    if patience_cnt == 0:
                        best_epoch = epoch
                    patience_cnt += 1
                    if patience_cnt == patience:
                        break
        print('\nTraining Finished.')

        # If early stopping, load best epoch
        if self.config["early_stopping"] == True:
            checkpoint = torch.load(f'weights/weights_{best_epoch}', map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        # Now run test
        test_loss, test_acc = self.test(val=False)
        log_dict = {'Test_Loss_{}'.format(model_type): test_loss,
                    'Test_Accuracy_{}'.format(model_type): test_acc}
        print(log_dict)
        if self.config["wandb"]:
            wandb.log(log_dict)

        print('\nTest Finished.')
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
