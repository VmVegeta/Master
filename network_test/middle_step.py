import multiprocessing
import socket
import json
import torch.nn as nn
from network_test.models import NnLayer
from network_test.pollution_dataset import get_dataset
import time
from network_test.tools import *
from torch.profiler import profile, record_function, ProfilerActivity


class MiddleModel(nn.Module):
    def __init__(self, output_size: int, device_count: int):
        super(MiddleModel, self).__init__()
        self.model = nn.Sequential(
            NnLayer(device_count * output_size, 64),
            NnLayer(64, 64),
            NnLayer(64, 64),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)
        self.early_exit = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.Linear(32, 1))

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h), self.early_exit(h)


class MiddleDevice:
    def __init__(self, station_id: str, output_size: int, device_count: int, address: str, server_address: str, port: int, server_port: int):
        self.station_id = station_id
        self.server_address = server_address
        self.address = address
        self.port = port
        self.server_port = server_port
        self.device_count = device_count

        self.model = MiddleModel(output_size, device_count)
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss_func = nn.MSELoss()
        self.binary_loss = nn.BCEWithLogitsLoss()

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

    def collect_data(self, server_socket, handle_function):
        thread_count = 0
        jobs = []
        parent_pipes = []
        while thread_count < self.device_count:
            client, address = server_socket.accept()
            print('Connected to: ' + address[0] + ':' + str(address[1]))
            parent_pipe, child_pipe = multiprocessing.Pipe()
            parent_pipes.append(parent_pipe)
            p = multiprocessing.Process(target=handle_function, args=(client, child_pipe))
            jobs.append(p)
            p.start()
            thread_count += 1
            print('Thread Number: ' + str(thread_count))

        data = [[]] * self.device_count
        for parent_pipe in parent_pipes:
            received = parent_pipe.recv()
            data[int(received[0])] = received[1]

        for proc in jobs:
            proc.join()
        return data

    def wait_and_train(self, server_socket, optimizer, true_value):
        train_data = self.collect_data(server_socket, train_client_thread)

        tensor_matrix = torch.tensor(train_data)
        tensor_matrix = torch.dequantize(tensor_matrix)

        epochs = 100
        outputs = []
        for i in range(epochs):
            matrix = tensor_matrix[:, i % 10, :]
            output = self.train(matrix, true_value, optimizer, i)
            if i >= epochs - 11:
                outputs.append(quantize_data(output))
        return outputs

    def wait_and_evaluate(self, server_socket, true_value):
        print('Eval Ready')
        train_data = self.collect_data(server_socket, train_client_thread)

        tensor_matrix = torch.tensor(train_data)
        tensor_matrix = torch.dequantize(tensor_matrix)
        output, predictions, ee = self.predict(tensor_matrix)
        self.print_results(predictions, true_value)
        return output

    def predict(self, X):
        self.model.eval()
        X = convert_tensor(X)
        return self.model(X)

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

    def create(self, train_true):
        print('Started Middle: ', self.station_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)

        data_dict = {"station_id": self.station_id, 'train': []}

        server_socket = socket.socket()
        try:
            server_socket.bind((self.address, self.port))
        except socket.error as e:
            print(str(e))
            exit(1)

        data_dict['train'] = self.wait_and_train(server_socket, optimizer, train_true)

        self.model.eval()
        output, predictions, ee_p = self.model(test_matrix)
        print("Eval loss:", self.loss_func(predictions, test_true))
        print("Eval R2:", r2_loss(predictions, test_true))

        self.send_data(data_dict)

        #print(prof.key_averages().table(sort_by="cpu_time_total"))
        print('Ended: ', self.station_id)

    def print_results(self, predictions, true_value):
        true_value = convert_tensor(true_value)
        loss = self.loss_func(predictions, true_value)
        print(loss)
        print(r2_loss(predictions, true_value))


def get_raw_data(connection):
    json_data = ''
    while True:
        data = connection.recv(26480000)
        if not data:
            break
        json_data += data.decode()
    data = json.loads(json_data)
    connection.close()
    return data


def train_client_thread(connection, child_pipe):
    data = get_raw_data(connection)
    station_id = data['station_id']
    child_pipe.send((station_id, data['train']))
    print('Thread done')


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset(history=6)
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    start_time = time.time()
    md = MiddleDevice('0', 16, '127.0.0.1', 10203)
    md.create(train_true)
    print('Time: ', time.time() - start_time)
