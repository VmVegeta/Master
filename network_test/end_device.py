import socket
import json
import torch.nn as nn
import torch
from network_test.models import RnnLayer, NnLayer
from torch.autograd import Variable
from network_test.pollution_dataset import get_dataset
#import time


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class DeviceModel(nn.Module):
    def __init__(self):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            RnnLayer(1, 32),
            #nn.Dropout(0.3),
            NnLayer(32, 64),
            #nn.Dropout(0.3),
            NnLayer(64, 32),
            #nn.Dropout(0.3),
            NnLayer(32, 16)
        )
        self.regression = nn.Linear(16, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h)


def train(model, train_matrix, train_true, optimizer, loss_func, last):
    model.train()
    if torch.cuda.is_available():
        train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
    data, target = Variable(train_matrix), Variable(train_true)

    optimizer.zero_grad()
    output, predictions = model(data)
    loss = loss_func(predictions, target).sum()
    loss.backward()
    optimizer.step()
    if last:
        print('{:.4f}'.format(loss))
        print('{:.4f}'.format(r2_loss(output, predictions)))

    return output


def create(station_id, train_matrix, train_true, cloud_address: str, port: int, epochs: int, test_matrix, test_true):
    print('Started: ' + str(station_id))
    s = socket.socket()

    #torch.manual_seed(1)
    model = DeviceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)

    data_dict = {"station_id": str(station_id), 'train': []}
    for epoch in range(1, epochs):
        last = epoch == epochs - 1
        output = train(model, train_matrix, train_true, optimizer, loss_func, last)
        #data_dict['train'].append(output.tolist())
        if epoch >= epochs - 11:
            data_dict['train'].append(output.tolist())

    output, predictions = model(test_matrix)
    print(loss_func(predictions, test_true))
    print(r2_loss(predictions, test_true))

    # data_dict['train'].append(output.tolist())
    try:
        s.connect((cloud_address, port))
        data = json.dumps(data_dict, separators=(',', ':'))
        encode = data.encode()
        #print(encode)
        s.sendall(encode)
        s.close()
    except socket.error as e:
        print(str(e))

    print('Ended: ', str(station_id))


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset(history=6)
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    create(0, train_ma, train_true, '127.0.0.1', 10203, 300, test_ma, test_true)
    #matrix = torch.tensor([[[1.0], [5.0], [9.0]], [[0.0], [10.0], [4.0]]])
    #true = torch.tensor([[1.0], [2.0]])
    #create(0, matrix, true, '192.168.0.104', 10203)
