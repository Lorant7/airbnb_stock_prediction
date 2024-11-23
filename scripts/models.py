import pytorch
from pytorch import nn



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.l = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        _ ,h_n = self.rnn(x)
        x = self.l(h_n)
        print("len of x: ", x.shape)
        return x