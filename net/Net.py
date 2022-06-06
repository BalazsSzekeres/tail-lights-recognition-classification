from typing import List

import numpy as np
import torch
import torch.nn as nn

from net.LSTM import LSTM


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict({
            # https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt
            # Calculate kernel shapes with https://madebyollin.github.io/convnet-calculator/
            # Kernel size and stride values are based on educated guesses
            # For now I simply assumed no padding
            # If we do add padding we can increase the kernel sizes: 1 extra padding -> 2 extra kernel size
            'cnn': nn.Sequential(
                # (3 x 277 x 277)
                nn.Conv2d(in_channels=self.config["input"]["dim"], out_channels=self.config["conv1"]["out"],
                          kernel_size=self.config["conv1"]["k"], stride=self.config["conv1"]["s"],
                          padding=self.config["conv1"]["p"]),
                nn.ReLU(),
                # (96 x 111 x 111)
                nn.MaxPool2d(kernel_size=self.config["pool1"]["k"], stride=self.config["pool1"]["s"],
                             padding=self.config["pool1"]["p"]),
                nn.BatchNorm2d(self.config["conv1"]["out"]),  # TODO: Do we want normalization ??
                # (96 x 55 x 55)
                nn.Conv2d(in_channels=self.config["conv1"]["out"], out_channels=self.config["conv2"]["out"],
                          kernel_size=self.config["conv2"]["k"], stride=self.config["conv2"]["s"],
                          padding=self.config["conv2"]["p"]),
                nn.ReLU(),
                # (384 x 26 x 26)
                nn.MaxPool2d(kernel_size=self.config["pool2"]["k"], stride=self.config["pool2"]["s"],
                             padding=self.config["pool2"]["p"]),
                nn.BatchNorm2d(self.config["conv2"]["out"]),  # TODO: Do we want normalization ??
                # (384 x 13 x 13)
                # For the last few layers I assume some paddings since the sizes don't change
                nn.Conv2d(in_channels=self.config["conv2"]["out"], out_channels=self.config["conv3"]["out"],
                          kernel_size=self.config["conv3"]["k"], stride=self.config["conv3"]["s"],
                          padding=self.config["conv3"]["p"]),
                nn.ReLU(),
                # (512 x 13 x 13)
                nn.Conv2d(in_channels=self.config["conv3"]["out"], out_channels=self.config["conv4"]["out"],
                          kernel_size=self.config["conv4"]["k"], stride=self.config["conv4"]["s"],
                          padding=self.config["conv4"]["p"]),
                nn.ReLU(),
                # (512 x 13 x 13)
                nn.Conv2d(in_channels=self.config["conv4"]["out"], out_channels=self.config["conv5"]["out"],
                          kernel_size=self.config["conv5"]["k"], stride=self.config["conv5"]["s"],
                          padding=self.config["conv5"]["p"]),
                nn.ReLU(),
                # (384 x 13 x 13)
                nn.Conv2d(in_channels=self.config["conv5"]["out"], out_channels=self.config["conv6"]["out"]["dim"],
                          kernel_size=self.config["conv6"]["k"], stride=self.config["conv6"]["s"],
                          padding=self.config["conv6"]["p"]),
                nn.ReLU(),
                # (384 x 13 x 13)
                nn.Flatten(),
                # (384 * 6 * 6 = 13824)
                nn.Linear(in_features=self.config["conv6"]["out"]["dim"] * self.config["conv6"]["out"]["size"] *
                                      self.config["conv6"]["out"]["size"], out_features=self.config["lstm"]["in"]),
                # (4096)
            ),
            'lstm': nn.LSTM(input_size=self.config["lstm"]["in"], hidden_size=self.config["lstm"]["hidden"], batch_first=True),
            # 'lstm': LSTM(input_size=self.config["lstm"]["in"], hidden_size=self.config["lstm"]["hidden"]),
            'out': nn.Sequential(
                nn.Linear(self.config["lstm"]["hidden"], self.config["output"]["size"]),
                nn.Softmax(dim=1)
            )
        })

    def forward(self, x):
        if isinstance(x, list):
            return self.forward_sequence_list(x)
        elif isinstance(x, torch.Tensor):
            """Forward an input through the CNN-LSTM network."""

            # First, pass the input through the CNN
            x = self.model['cnn'](x)

            # Then, reshape the input and pass it through the LSTM
            x = x.view(x.shape[0], 1, -1)
            x = self.model['lstm'](x)[0][-1]

            # From the paper:
            # "The output of each LSTM layer is sent to the last fully connected layer of our network
            # to compute a class probability for each time step"
            y = self.model['out'](x)[0]

            # "Given a test frame, instead of taking average among the output predictions, we take the
            # prediction of the last frame as the label for the entire input sequence."
            return y

    def forward_sequence_list(self, x_list: List[torch.Tensor]):
        """Forward a list of sequences"""
        y = torch.stack([self.forward(x) for x in x_list])
        return y
