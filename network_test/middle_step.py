import multiprocessing
from network_test.models import NnLayer
from network_test.pollution_dataset import get_dataset
import time
from client_base import *


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


class MiddleDevice(ClientBase):
    def __init__(self, station_id: str, output_size: int, device_count: int, address: str, server_address: str, port: int, server_port: int, ee_range=8):
        model = MiddleModel(output_size, device_count)
        super().__init__(model, station_id, output_size, server_address, server_port, ee_range)
        self.address = address
        self.port = port
        self.device_count = device_count

    def collect_data(self, server_socket, handle_function):
        thread_count = 0
        jobs = []
        parent_pipes = []
        while thread_count < self.device_count:
            client, address = server_socket.accept()
            print('M Connected to: ' + address[0] + ':' + str(address[1]))
            parent_pipe, child_pipe = multiprocessing.Pipe()
            parent_pipes.append(parent_pipe)
            p = multiprocessing.Process(target=handle_function, args=(client, child_pipe))
            jobs.append(p)
            p.start()
            thread_count += 1
            print('M Thread Number: ' + str(thread_count))

        data = [None] * self.device_count
        for parent_pipe in parent_pipes:
            received = parent_pipe.recv()
            tensor_matrix = torch.tensor(received[1], dtype=torch.float32)
            #tensor_matrix = torch.dequantize(tensor_matrix)
            data[int(received[0])] = tensor_matrix

        for proc in jobs:
            proc.join()
        return data

    def wait_and_train(self, server_socket, optimizer, true_value):
        train_data = self.collect_data(server_socket, train_client_thread)

        train_data = torch.concat(train_data, 2)
        epochs = 300
        outputs = []

        X = train_data[0]
        X, y = convert_tensor(X), convert_tensor(true_value)

        for i in range(epochs):
            is_last = epochs - 1 == i
            #matrix = tensor_matrix[:, i % 10, :]
            self.train(X, y, optimizer, is_last)

        for i in range(epochs * 2):
            is_last = epochs * 2 - 1 == i
            output, loss = self.train(X, y, optimizer, is_last)
            if is_last:
                outputs.append(quantize_data(output))
        return outputs

    def start_eval(self, server_socket, true_value):
        print('Eval Ready')
        data = self.collect_data(server_socket, train_client_thread)
        data = torch.concat(data, 1)
        #tensor_matrix = torch.tensor(data)
        #tensor_matrix = torch.dequantize(data)
        output, predictions, ee = self.predict(data)
        self.print_results(predictions, true_value)
        return quantize_data(output)

    def predict(self, X):
        self.model.eval()
        X = convert_tensor(X)
        return self.model(X)

    def create(self, train_true, test_true):
        print('Started Middle: ', self.station_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.008)
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)

        data_dict = {"station_id": self.station_id, 'train': []}

        server_socket = socket.socket()
        try:
            server_socket.bind((self.address, self.port))
            server_socket.listen(self.device_count)
        except socket.error as e:
            print(str(e))
            exit(1)

        data_dict['train'] = self.wait_and_train(server_socket, optimizer, train_true)
        self.send_data(data_dict)
        print('M Ended: ', self.station_id)

        data_dict['train'] = self.start_eval(server_socket, test_true)
        self.send_data(data_dict)

    def print_results(self, predictions, true_value):
        true_value = convert_tensor(true_value)
        loss = self.loss_func(predictions, true_value)
        print('{:.4f}'.format(loss))
        print('{:.4f}'.format(r2_loss(predictions, true_value)))


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
    start_time = time.time()
    md = MiddleDevice('2', 16, 3, '', '127.0.0.1', 13203, 10203)
    md.create(train_true, test_true)
    print('Time: ', time.time() - start_time)
