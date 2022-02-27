import socket
import json
import torch.nn as nn
import torch
from network_test.models import RnnLayer, NnLayer
from torch.autograd import Variable


class DeviceModel(nn.Module):
    def __init__(self):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            RnnLayer(1, 32),
            NnLayer(32, 32),
            NnLayer(32, 16)
        )
        self.regression = nn.Linear(16, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h)


def train(model, train_matrix, train_true, optimizer, loss_func):
    model.train()
    if torch.cuda.is_available():
        train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
    data, target = Variable(train_matrix), Variable(train_true)

    optimizer.zero_grad()
    output, predictions = model(data)
    loss = loss_func(predictions, target)
    loss.backward()
    optimizer.step()

    return output


def create(station_id, train_matrix, train_true, cloud_ip: str, port: int):
    epochs = 10
    s = socket.socket()

    model = DeviceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        s.connect(('127.0.0.1', port))
    except socket.error as e:
        print(str(e))

    data_dict = {"station_id": str(station_id), 'train': []}

    for epoch in range(1, epochs):
        print(epoch)
        output = train(model, train_matrix, train_true, optimizer, loss_func)
        data_dict['train'].append(output.tolist())

    data = json.dumps(data_dict)
    s.sendall(data.encode())
    s.close()


if __name__ == "__main__":
    matrix = torch.tensor([[[1.0], [5.0], [9.0]], [[0.0], [10.0], [4.0]]])
    true = torch.tensor([[1.0], [2.0]])
    create(0, matrix, true, '192.168.0.104', 10203)
