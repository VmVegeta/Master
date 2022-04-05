import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudModule(nn.Module):
    def __init__(self, num_devices: int, output_size: int):
        super(CloudModule, self).__init__()
        self.num_devices = num_devices
        #self.device_models = []
        #for _ in range(num_devices):
        #    self.device_models.append(DeviceModel(in_channels))
        #self.device_models = nn.ModuleList(self.device_models)

        cloud_input_channels = output_size * num_devices
        self.cloud_model = nn.Sequential(
            NnLayer(cloud_input_channels, 128),
            NnLayer(128, 128),
            NnLayer(128, 128),
            NnLayer(128, 128),
            NnLayer(128, 1)
        )

    def forward(self, x):
        """
        first_dim = x.shape[1]
        second_dim = x.shape[0] * x.shape[2]
        x = torch.permute(x, (1, 0, 2))
        x = torch.reshape(x, (first_dim, second_dim))
        """
        return self.cloud_model(x)


class RnnLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RnnLayer, self).__init__()
        self.hidden_dim = out_channels
        self.n_layers = 2

        self.rnn = nn.LSTM(in_channels, out_channels, self.n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_dim, out_channels)

    def forward(self, x):
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

