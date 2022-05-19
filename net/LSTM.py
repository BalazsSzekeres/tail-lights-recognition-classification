import torch
import torch.nn as nn
import math


class LSTM(nn.Module):
    """
    Long short-term memory recurrent unit which has the following update rule:
        it ​= σ(W_xi * ​xt ​+ b_xi ​+ W_hi * ​h(t−1) ​+ b_hi​)
        ft​ = σ(W_xf * ​xt ​+ b_xf ​+ W_hf * ​h(t−1) ​+ b_hf​)
        gt ​= tanh(W_xg * ​xt ​+ b_xg ​+ W_hg * ​h(t−1) ​+ b_hg​)
        ot ​= σ(W_xo * ​xt ​+ b_xo​ + W_ho ​h(t−1) ​+ b_ho​)
        ct ​= ft​ ⊙ c(t−1) ​+ it ​⊙ gt​
        ht ​= ot​ ⊙ tanh(ct​)​
    """

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # Input to hidden weights
        self.weight_xh = None

        # Hidden to hidden biases
        self.weight_hh = None

        # Input to hidden biases
        self.bias_xh = None

        # Hidden to hidden biases
        self.bias_hh = None

        # LSTM parameters
        self.weight_xh = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias_xh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        """
        Initialize network parameters.
        """

        std = 1.0 / math.sqrt(self.hidden_size)
        self.weight_xh.data.uniform_(-std, std)
        self.weight_hh.data.uniform_(-std, std)
        self.bias_xh.data.uniform_(-std, std)
        self.bias_hh.data.uniform_(-std, std)

    def forward(self, x):
        """
        Args:
            x: input with shape (n, tt, D) where n is number of samples, tt is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (n, tt, h) where h is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (tt, n, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        tt, n, h = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (n, h)
        h0 = torch.zeros(n, h, device=x.device)
        c0 = torch.zeros(n, h, device=x.device)

        # Define a list to store outputs. We will then stack them.
        y = []

        ht_1 = h0
        ct_1 = c0
        for t in range(tt):
            # LSTM update rule
            xh = torch.addmm(self.bias_xh, x[t], self.weight_xh)
            hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            it = torch.sigmoid(xh[:, 0:h] + hh[:, 0:h])
            ft = torch.sigmoid(xh[:, h:2 * h] + hh[:, h:2 * h])
            gt = torch.tanh(xh[:, 2 * h:3 * h] + hh[:, 2 * h:3 * h])
            ot = torch.sigmoid(xh[:, 3 * h:4 * h] + hh[:, 3 * h:4 * h])
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        # Stack the outputs. After this operation, output will have shape of
        # (tt, n, h)
        y = torch.stack(y)

        # Switch time and batch dimension, (tt, n, h) -> (n, tt, h)
        y = y.transpose(0, 1)
        return y
