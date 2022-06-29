from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn


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


class DeviceModel(nn.Module):
    def __init__(self, in_channels, output_size):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            RnnLayer(in_channels, 16),
            NnLayer(16, 64),
            NnLayer(64, 64),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h)


class DeviceModelNoRnn(nn.Module):
    def __init__(self, in_channels, output_size):
        super(DeviceModelNoRnn, self).__init__()
        self.model = nn.Sequential(
            NnLayer(in_channels, 64),
            NnLayer(64, 64),
            NnLayer(64, 64),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h)


class MiddleModel(nn.Module):
    def __init__(self, in_channels, output_size):
        super(MiddleModel, self).__init__()
        self.model = nn.Sequential(
            NnLayer(in_channels, 64),
            NnLayer(64, 64),
            NnLayer(64, 64),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h)


class DDNN(nn.Module):
    def __init__(self, in_channels, num_devices, is_middle=True, output_size=8):
        super(DDNN, self).__init__()
        self.is_middle = is_middle
        self.num_devices = num_devices
        self.device_models = []
        for _ in range(num_devices):
            self.device_models.append(DeviceModel(in_channels, output_size))
        self.device_models = nn.ModuleList(self.device_models)

        if is_middle:
            self.middle_models = [
                MiddleModel(output_size * 2, output_size),
                MiddleModel(output_size * 2, output_size),
                MiddleModel(output_size * 3, output_size)
            ]
            self.middle_models = nn.ModuleList(self.middle_models)

            cloud_input_channels = output_size * 3 + 5
        else:
            cloud_input_channels = output_size * num_devices

        self.cloud_model = nn.Sequential(
            NnLayer(cloud_input_channels, 128),
            NnLayer(128, 128),
            NnLayer(128, 128),
            NnLayer(128, 128)
        )
        self.regression = nn.Linear(128, 1)

    def forward(self, x, later_input):
        hs, predictions = [], []
        for i, device_model in enumerate(self.device_models):
            h, prediction = device_model(x[:, i])
            hs.append(h)
            predictions.append(prediction)

        if self.is_middle:
            inputs = [torch.cat(hs[0:2], dim=1), torch.cat(hs[2:4], dim=1), torch.cat(hs[4:7], dim=1)]
            hs = []
            for i, middle_model in enumerate(self.middle_models):
                h, prediction = middle_model(inputs[i])
                hs.append(h)
                predictions.append(prediction)
        hs.append(later_input)
        h = torch.cat(hs, dim=1)
        h = self.cloud_model(h)
        prediction = self.regression(h)
        predictions.append(prediction)
        return predictions
