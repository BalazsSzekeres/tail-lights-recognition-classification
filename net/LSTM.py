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

        ########################################################################
        #    TODO: Create weight and bias tensors with given name above with   #
        #                             correct sizes.                           #
        # NOTE: Don't forget to encapsulate weights and biases in nn.Parameter #
        ########################################################################

        # LSTM parameters
        self.weight_xh = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias_xh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

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
            x: input with shape (N, T, D) where N is number of samples, T is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T, N, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device)
        c0 = torch.zeros(N, H, device=x.device)

        # Define a list to store outputs. We will then stack them.
        y = []

        ########################################################################
        #                 TODO: Implement forward pass of LSTM                 #
        ########################################################################

        ht_1 = h0
        ct_1 = c0
        for t in range(T):
            # LSTM update rule
            xh = torch.addmm(self.bias_xh, x[t], self.weight_xh)
            hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            it = torch.sigmoid(xh[:, 0:H] + hh[:, 0:H])
            ft = torch.sigmoid(xh[:, H:2 * H] + hh[:, H:2 * H])
            gt = torch.tanh(xh[:, 2 * H:3 * H] + hh[:, 2 * H:3 * H])
            ot = torch.sigmoid(xh[:, 3 * H:4 * H] + hh[:, 3 * H:4 * H])
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        # Stack the outputs. After this operation, output will have shape of
        # (T, N, H)
        y = torch.stack(y)

        # Switch time and batch dimension, (T, N, H) -> (N, T, H)
        y = y.transpose(0, 1)
        return y
