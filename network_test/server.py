import socket
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

from network_test.models import NnLayer
from threading import Thread

def on_new_client(clientsocket, addr):
    while True:
        msg = clientsocket.recv(1024)
        #do some checks and if msg == someWeirdSignal: break:
        print(addr, ' >> ', msg)
        #msg = raw_input('SERVER >> ')
        #Maybe some code to compute the last digit of PI, play game or anything else can go here and when you are done.
        clientsocket.send(msg)
    clientsocket.close()

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
        #hs, predictions = [], []
        #for i, device_model in enumerate(self.device_models):
        #    h, prediction = device_model(x[:, i])
        #    hs.append(h)
        #    predictions.append(prediction)

        #h = torch.cat(x, dim=1) # This one is needed
        h = self.cloud_model(x)
        return self.regression(h)


def train(model, train_matrix, train_true, optimizer, loss_func):
    model.train()
    if torch.cuda.is_available():
        train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
    data, target = Variable(train_matrix), Variable(train_true)

    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_func(predictions, target)
    loss.backward()
    optimizer.step()
    print(predictions)

    return predictions


def main(true_value):
    model = CloudModule(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    s = socket.socket()
    port = 10203
    s.bind(('', port))# 192.168.0.104
    s.listen(5)
    print('Ready')
    while True:
        c, addr = s.accept()
        with c:
            while True:
                # print('Got connection from', addr)
                # c.send(b'Thank you for connecting')
                data = c.recv(140 * 16000)
                if not data:
                    break
                decoded = data.decode()
                data = json.loads(decoded)
                train(model, torch.tensor(data["train"]), true_value, optimizer, loss_func)
        result = model(torch.tensor([0.0, 0.27625516057014465, 0.0, 0.23540158569812775, 0.3637048602104187, 0.0, 0.24097499251365662, 0.0, 0.16266408562660217, 0.0, 0.0, 0.0, 0.4259628355503082, 0.2792994976043701, 0.0, 0.002364151179790497]))
        print(result)
    # c.close()

    while True:
        client, address = s.accept()  # Establish connection with client.
        Thread(target=on_new_client, args=(client, address))
        # Note it's (addr,) not (addr) because second parameter is a tuple
        # Edit: (c,addr)
        # that's how you pass arguments to functions when creating new threads using thread module.
    s.close()


if __name__ == '__main__':
    true = torch.tensor([[1.0], [2.0]])
    main(true)
