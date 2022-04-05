import torch.nn as nn
import socket
import json
from network_test.tools import *


class ClientBase:
    def __init__(self, model: nn.Module, station_id: str, output_size: int, server_address: str,  server_port: int, ee_range=8):
        self.station_id = station_id
        self.server_address = server_address
        self.output_size = output_size
        self.server_port = server_port

        self.loss_func = nn.MSELoss()
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.model = model
        self.ee_range = ee_range
        if torch.cuda.is_available():
            self.model.cuda()

    def send_data(self, data_dict):
        s = socket.socket()
        try:
            s.connect((self.server_address, self.server_port))
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
        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(X)

        loss = self.loss_func(predictions, y)
        loss.backward()
        optimizer.step()
        if last:
            print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(predictions, y)))

        return output, loss

    def early_exit_train(self, X, y, optimizer, last):
        self.model.train()
        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(X)
        to_exit = early_exit(predictions, y, self.ee_range)
        loss = self.loss_func(predictions, y)
        exit_loss = self.binary_loss(to_exit_prediction, to_exit)
        combined_loss = exit_loss + loss
        combined_loss.backward()
        optimizer.step()

        if last:
            print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(predictions, y)))

        return output, loss

    def evaluate(self, test_matrix):
        print('Evaluate: ', self.station_id)
        self.model.eval()
        test_matrix = convert_tensor(test_matrix)
        output, predictions, ee_p = self.model(test_matrix)
        data_dict = {"station_id": self.station_id, 'train': quantize_data(output)}
        self.send_data(data_dict)
