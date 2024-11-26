import pytorch
from pytorch import nn
import torch.nn.init as init

#############################################################
####                                                     ####
####                       RNN                           ####
####                                                     ####
#############################################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.l = nn.Linear(in_features=hidden_size, out_features=output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize RNN weights using Xavier initialization
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)  # Input-hidden weights
            elif 'weight_hh' in name:
                init.orthogonal_(param)  # Hidden-hidden weights
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zeros

        # Initialize the linear layer with Xavier initialization
        init.xavier_uniform_(self.l.weight)
        if self.l.bias is not None:
            init.zeros_(self.l.bias)

    def forward(self, x):
        x = x.unsqueeze(0)
        output, h_n = self.rnn(x)
        print(output.shape)
        x = self.l(h_n[-1]).squeeze()
        # No need to pass the logits of the final linear layer through a sigmoid function because I am using
        # the BCEWithLogitsLoss which applies it internally
        return x


#############################################################
####                                                     ####
####                        LSTM                         ####
####                                                     ####
#############################################################


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fll = nn.Linear(in_features=hidden_size, out_features=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize RNN weights using Xavier initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)  # Input-hidden weights
            elif 'weight_hh' in name:
                init.orthogonal_(param)  # Hidden-hidden weights
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zeros

        # Initialize the linear layer with Xavier initialization
        init.xavier_uniform_(self.fll.weight)
        if self.fll.bias is not None:
            init.zeros_(self.fll.bias)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.lstm(x)
        x = self.fll(x[1][-1])
        return x.squeeze(-1)[-1]

