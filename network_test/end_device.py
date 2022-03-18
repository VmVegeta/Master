import socket
import json
import torch.nn as nn
from network_test.models import RnnLayer, NnLayer
from network_test.pollution_dataset import get_dataset
import time
from network_test.tools import *
from torch.profiler import profile, record_function, ProfilerActivity


class DeviceModel(nn.Module):
    def __init__(self, output_size: int):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            RnnLayer(1, 32),
            #nn.Dropout(0.3),
            NnLayer(32, 64),
            #nn.Dropout(0.3),
            NnLayer(64, 64),
            #nn.Dropout(0.3),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)
        self.early_exit = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.Linear(32, 1))

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h), self.early_exit(h)


class EndDevice:
    def __init__(self, station_id: str, output_size: int, server_address: str, port: int):
        self.station_id = station_id
        self.server_address = server_address
        self.port = port
        torch.manual_seed(1)

        self.model = DeviceModel(output_size)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss_func = nn.MSELoss()
        self.binary_loss = nn.BCEWithLogitsLoss()

    def send_data(self, data_dict):
        s = socket.socket()
        try:
            s.connect((self.server_address, self.port))
            data = json.dumps(data_dict, separators=(',', ':'))
            encode = data.encode()
            print(len(encode))
            s.sendall(encode)
            s.close()
        except socket.error as e:
            print(str(e))
        s.close()

    def train(self, X, y, optimizer, last):
        self.model.train()
        X, y = convert_tensor(X), convert_tensor(y)

        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(X)

        loss = self.loss_func(predictions, y)
        loss.backward()
        optimizer.step()
        if last:
            print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(predictions, y)))

        return output

    def early_exit_train(self, X, y, optimizer, last):
        self.model.train()
        X, y = convert_tensor(X), convert_tensor(y)

        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(X)
        to_exit = early_exit(predictions, y, 10)
        exit_loss = self.binary_loss(to_exit_prediction, to_exit)
        exit_loss.backward()
        optimizer.step()

        if last:
            #print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(predictions, y)))

        return output

    def evaluate(self, test_matrix):
        print('Evaluate: ', self.station_id)
        self.model.eval()
        output, predictions, ee_p = self.model(test_matrix)
        data_dict = {"station_id": self.station_id, 'train': quantize_data(output)}
        self.send_data(data_dict)

    def create(self, epochs, train_matrix, train_true, test_matrix, test_true):
        print('Started: ', self.station_id)
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.LSTM}, dtype=torch.float16)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)

        data_dict = {"station_id": self.station_id, 'train': []}
        #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #with record_function("model_train"):
        for epoch in range(1, int(epochs)):
            last = epoch == epochs - 1
            output = self.train(train_matrix, train_true, optimizer, last)

            #data_dict['train'].append(output.tolist())
            if epoch >= epochs - 11:
                data_dict['train'].append(quantize_data(output))
                #data_dict['train'].append(shorten_data(output))

        self.model.eval()
        output, predictions, ee_p = self.model(test_matrix)
        print("Eval loss:", self.loss_func(predictions, test_true))
        #print("Eval R2:", r2_loss(predictions, test_true))

        # Try with two output then predict both
        #ee = early_exit(predictions, test_true, 5)
        #print("Binary Acc:", binary_acc(ee_p, ee))

        if False:
            for epoch in range(1, int(epochs)):
                last = epoch == epochs - 1
                if epoch % 4 == 0:
                    self.train(train_matrix, train_true, optimizer, last)
                self.early_exit_train(train_matrix, train_true, optimizer, last)
                #data_dict['train'].append(output.tolist())
                #if epoch >= epochs - 11:
                #data_dict['train'].append()

            self.model.eval()
            output, predictions, ee_p = self.model(test_matrix)
            print("Eval loss:", self.loss_func(predictions, test_true))
            print("Eval R2:", r2_loss(predictions, test_true))

            # Try with two output then predict both
            ee = early_exit(predictions, test_true, 10)
            print("Binary Acc:", binary_acc(ee_p, ee))
            print(torch.count_nonzero(torch.round(torch.sigmoid(ee_p))))

        self.send_data(data_dict)

        #print(prof.key_averages().table(sort_by="cpu_time_total"))
        print('Ended: ', self.station_id)


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset(history=6)
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    start_time = time.time()
    ed = EndDevice('0', 16, '127.0.0.1', 10203)
    ed.create(10, train_ma, train_true, test_ma, test_true)
    print('Time: ', time.time() - start_time)
