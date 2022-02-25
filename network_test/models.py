import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RnnLayer, self).__init__()
        self.hidden_dim = 32
        self.n_layers = 2

        self.rnn = nn.RNN(in_channels, 32, self.n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_dim, out_channels)

    def forward(self, x):
        hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, hidden = self.rnn(x, hidden.detach())

        out = out[:, -1, :]
        out = self.fc(out)

        return out


class NnLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NnLayer, self).__init__()
        self.layer = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.layer(x))

