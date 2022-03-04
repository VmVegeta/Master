import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudModule(nn.Module):
    def __init__(self, num_devices):
        super(CloudModule, self).__init__()
        self.num_devices = num_devices
        #self.device_models = []
        #for _ in range(num_devices):
        #    self.device_models.append(DeviceModel(in_channels))
        #self.device_models = nn.ModuleList(self.device_models)

        cloud_input_channels = 16 * num_devices
        self.cloud_model = nn.Sequential(
            NnLayer(cloud_input_channels, 64),
            NnLayer(64, 64),
            # NnLayer(64, 64),
            NnLayer(64, 128)
        )
        self.regression = nn.Linear(128, 1)

    def forward(self, x):
        """
        hs, predictions = [], []
        for i, device_model in enumerate(self.device_models):
            h, prediction = device_model(x[:, i])
            hs.append(h)
            predictions.append(prediction)

        h = torch.cat(x, dim=1) # This one is needed
        """
        first_dim = x.shape[1]
        second_dim = x.shape[0] * x.shape[2]
        x = torch.permute(x, (1, 0, 2))
        x = torch.reshape(x, (first_dim, second_dim))
        h = self.cloud_model(x)
        return self.regression(h)


class RnnLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RnnLayer, self).__init__()
        self.hidden_dim = 32
        self.n_layers = 2

        self.rnn = nn.LSTM(in_channels, 32, self.n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_dim, out_channels)

    def forward(self, x):
        #hidden = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, hidden = self.rnn(x)

        out = out[:, -1, :]
        out = self.fc(out)

        return out


class NnLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NnLayer, self).__init__()
        self.layer = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.layer(x))

