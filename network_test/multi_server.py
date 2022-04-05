import multiprocessing
import socket
import json

import torch

from network_test.models import CloudModule
from network_test.pollution_dataset import get_dataset
from network_test.tools import *


class MultiServer:
    def __init__(self, device_count: int, output_size: int, host='', port=10203):
        self.device_count = device_count
        self.model = CloudModule(device_count, output_size)
        if torch.cuda.is_available():
            self.model.cuda()
        self.host = host
        self.port = port

    def launch(self, true_value, test_true):
        server_socket = socket.socket()
        try:
            server_socket.bind((self.host, self.port))
        except socket.error as e:
            print(str(e))
            exit(1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.008)
        print('Ready to connect')
        server_socket.listen(self.device_count)
        self.wait_and_train(server_socket, optimizer, torch.nn.MSELoss(), true_value)
        # self.start_eval(server_socket, test_true)
        server_socket.close()
        print("Socket Close")

    def wait_and_train(self, server_socket, optimizer, loss_func, true_value):
        train_data = self.collect_data(server_socket, train_client_thread)

        train_data = torch.concat(train_data, 2)
        true_value = convert_tensor(true_value)
        train_data = convert_tensor(train_data)
        epochs = 300
        # for i in range(tensor_matrix.shape[0]):
        # matrix = tensor_matrix[i % 10]
        for i in range(epochs):
            matrix = train_data[0]
            to_print = i == epochs - 1
            self.train(matrix, true_value, optimizer, loss_func, to_print)

    def train(self, X, y, optimizer, loss_func, to_print):
        self.model.train()

        optimizer.zero_grad()
        predictions = self.model(X)
        loss = loss_func(predictions, y)
        loss.backward()
        optimizer.step()

        if to_print:
            print('{:.4f}'.format(loss))
            print('{:.4f}'.format(r2_loss(predictions, y)))

        return predictions

    def start_eval(self, server_socket, true_value):
        print('Eval Ready')
        data = self.collect_data(server_socket, train_client_thread)
        data = torch.concat(data, 1)
        predictions = self.predict(data)
        print_results(predictions, true_value, torch.nn.MSELoss())

    def predict(self, X):
        self.model.eval()
        X = convert_tensor(X)
        return self.model(X)

    def start_inference(self, server_socket):
        print('Inference Ready')
        while True:
            data = self.collect_data(server_socket, infer_client_thread)
            datetime = None
            """
            train_data = [[]] * num_devices
            for parent_pipe in parent_pipes:
                received = parent_pipe.recv()
                train_data[int(received[0])] = received[1]
                if datetime is None:
                    datetime = received[2]
                else:
                    if datetime != received[2]:
                        print("Datetimes do not match")
                        print("Program just gave up")
                        exit(1)

                print("Works: ", received[0])

            for proc in jobs:
                proc.join()
            """
            tensor_matrix = torch.tensor(data)
            tensor_matrix = torch.dequantize(tensor_matrix)
            prediction = self.predict(tensor_matrix)
            print('Predicted value for Bygdø Alle ', datetime, '{:.4f}'.format(prediction))

    def collect_data(self, server_socket, handle_function):
        thread_count = 0
        jobs = []
        parent_pipes = []
        while thread_count < self.device_count:
            client, address = server_socket.accept()
            #print('Connected to: ' + address[0] + ':' + str(address[1]))
            parent_pipe, child_pipe = multiprocessing.Pipe()
            parent_pipes.append(parent_pipe)
            p = multiprocessing.Process(target=handle_function, args=(client, child_pipe))
            jobs.append(p)
            p.start()
            thread_count += 1
            #print('Thread Number: ' + str(thread_count))

        data = [None] * self.device_count
        for parent_pipe in parent_pipes:
            received = parent_pipe.recv()
            tensor_matrix = torch.tensor(received[1], dtype=torch.int8)
            tensor_matrix = torch.dequantize(tensor_matrix)
            data[int(received[0])] = tensor_matrix

        for proc in jobs:
            proc.join()
        return data


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
    #print('Thread done')


def infer_client_thread(connection, child_pipe):
    data = get_raw_data(connection)
    station_id = data['station_id']
    child_pipe.send((station_id, data['infer'], data['datetime']))


def print_results(predictions, true_value, loss_func):
    true_value = convert_tensor(true_value)
    loss = loss_func(predictions, true_value)
    print('{:.4f}'.format(loss))
    print('{:.4f}'.format(r2_loss(predictions, true_value)))


if __name__ == '__main__':
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    device_count = 7
    server = MultiServer(device_count, 16)
    server.launch(train_true, test_true)
    """
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    device_count = 7
    output_sizes = [16, 32]
    servers = []
    for output_size in output_sizes:
        print(str(output_size))
        port = 10000 + output_size
        server = MultiServer(device_count, output_size, port=port)
        server.launch(train_true, test_true)
        servers.append(server)

        s = socket.socket()
        try:
            s.connect(('127.0.0.1', 12122))
            s.sendall(b'next')
            s.close()
        except socket.error as e:
            print(str(e))
        s.close()

    for server in servers:
        server_socket = socket.socket()
        try:
            server_socket.bind((server.host, server.port))
        except socket.error as e:
            print(str(e))
            exit(1)
        server_socket.listen(device_count)
        server.start_eval(server_socket, test_true)
    """
