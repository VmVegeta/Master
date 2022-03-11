import socket
import json
import torch.nn as nn
import torch
from network_test.models import RnnLayer, NnLayer
from torch.autograd import Variable
import torch.nn.functional as F
from network_test.pollution_dataset import get_dataset
import time
import math


def quantize_data(output: torch.tensor):
    return torch.quantize_per_tensor(output, 0.1, 0, torch.quint8).int_repr().tolist()


def shorten_data(output: torch.tensor):
    return [[math.floor(x * 100) / 100.0 for x in row] for row in output.tolist()]


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def binary_acc(y_pred, y_test):
    sigmoidScores = torch.sigmoid(y_pred)
    #softmaxFunc = nn.Softmax(dim=1)
    #softmaxScores = softmaxFunc(y_pred)
    # y_pred_tag = torch.round(test)
    y_pred_tag = sigmoidScores > 0.5

    #correct_results_sum = (y_pred_tag == y_test).sum()
    correct_results_sum = ((y_pred_tag == True) & (y_test == 1.)).sum()
    #acc = correct_results_sum / y_test.shape[0]
    total_prediction = (y_pred_tag == True).sum()
    print("Total EE: ", total_prediction)
    acc = correct_results_sum / total_prediction
    acc = torch.round(acc * 10000) / 100

    return acc


def early_exit(prediction: torch.Tensor, target: torch.Tensor, acceptable_range: float):
    difference = torch.sub(prediction, target)
    difference = torch.abs(difference)
    return torch.where(difference < acceptable_range, 1.0, 0.0)


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
        self.early_exit = nn.Linear(16, 1)

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h), self.early_exit(h)


class EndDevice:
    def __init__(self, station_id: str, server_address: str, port: int):
        self.station_id = station_id
        self.server_address = server_address
        self.port = port
        torch.manual_seed(1)
        self.model = DeviceModel()

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

    def train(self, train_matrix, train_true, optimizer, loss_func, last):
        self.model.train()
        if torch.cuda.is_available():
            train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
        data, target = Variable(train_matrix), Variable(train_true)

        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(data)

        loss = loss_func(predictions, target)
        loss.backward()
        optimizer.step()
        if last:
            print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(output, predictions)))

        return output

    def early_exit_train(self, train_matrix, train_true, optimizer, loss_func, last):
        self.model.train()
        if torch.cuda.is_available():
            train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
        data, target = Variable(train_matrix), Variable(train_true)

        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(data)

        to_exit = early_exit(predictions, target, 5)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        exit_loss = criterion(to_exit_prediction, to_exit)
        #print('EE loss: ', '{:.4f}'.format(exit_loss))
        #loss = loss_func(predictions, target) + exit_loss
        exit_loss.backward()
        optimizer.step()
        if last:
            #print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(output, predictions)))

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
        loss_func = torch.nn.MSELoss()
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)

        data_dict = {"station_id": self.station_id, 'train': []}
        for epoch in range(1, epochs):
            last = epoch == epochs - 1
            output = self.train(train_matrix, train_true, optimizer, loss_func, last)

            #data_dict['train'].append(output.tolist())
            if epoch >= epochs - 11:
                data_dict['train'].append(quantize_data(output))
                #data_dict['train'].append(shorten_data(output))

        self.model.eval()
        output, predictions, ee_p = self.model(test_matrix)
        print("Eval loss:", loss_func(predictions, test_true))
        print("Eval R2:", r2_loss(predictions, test_true))

        # Try with two output then predict both
        ee = early_exit(predictions, test_true, 5)
        print("Binary Acc:", binary_acc(ee_p, ee))

        if False:
            for epoch in range(1, int(epochs / 2)):
                last = epoch == epochs - 1
                output = self.early_exit_train(train_matrix, train_true, optimizer, loss_func, last)
                #data_dict['train'].append(output.tolist())
                if epoch >= epochs - 11:
                    data_dict['train'].append([[math.floor(x * 100) / 100.0 for x in row] for row in output.tolist()])

            self.model.eval()
            output, predictions, ee_p = self.model(test_matrix)
            print("Eval loss:", loss_func(predictions, test_true))
            print("Eval R2:", r2_loss(predictions, test_true))

            # Try with two output then predict both
            ee = early_exit(predictions, test_true, 5)
            print("Binary Acc:", binary_acc(ee_p, ee))
            print(torch.count_nonzero(torch.round(torch.sigmoid(ee_p))))

        # data_dict['train'].append(output.tolist())
        self.send_data(data_dict)

        print('Ended: ', self.station_id)


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset(history=6)
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    start_time = time.time()
    ed = EndDevice('0', '127.0.0.1', 10203)
    ed.create(10, train_ma, train_true, test_ma, test_true)
    print('Time: ', time.time() - start_time)
    #matrix = torch.tensor([[[1.0], [5.0], [9.0]], [[0.0], [10.0], [4.0]]])
    #true = torch.tensor([[1.0], [2.0]])
    #create(0, matrix, true, '192.168.0.104', 10203)
