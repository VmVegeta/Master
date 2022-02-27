import socket
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

from network_test.models import CloudModule


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
