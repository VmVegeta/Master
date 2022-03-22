from network_test.end_device import EndDevice
from _thread import start_new_thread
from network_test.pollution_dataset import get_dataset
import torch
import socket


def launch_end_device(end_device: EndDevice, epochs: int, train_matrix: torch.Tensor, train_ture: torch.Tensor,
                      test_matrix, test_true):
    end_device.create(epochs, train_matrix, train_ture, test_matrix, test_true)


def start_evaluation(end_device: EndDevice, test_matrix: torch.Tensor):
    end_device.evaluate(test_matrix)


def start_launches(num_devices, output_size, end_devices, train_matrix, test_matrix, train_true, test_true, epochs):
    for device_id in range(num_devices):
        end_device = EndDevice(str(device_id), output_size, '127.0.0.1', 10200 + output_size)
        end_devices.append(end_device)
        device_train_matrix = train_matrix[:, device_id]
        device_test_matrix = test_matrix[:, device_id]

        start_new_thread(
            launch_end_device,
            (end_device, epochs, device_train_matrix, train_true, device_test_matrix, test_true)
        )


def main(epochs: int, output_size: int):
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = train_matrix.shape[1]

    end_devices = []
    output_sizes = [8, 16, 32]
    i = 0
    start_launches(num_devices, output_sizes[i], end_devices, train_matrix, test_matrix, train_true, test_true, epochs)

    while i < 2:
        """
        prompt = input('"exit"/"eval": ')
        if prompt == 'exit':
            exit(-1)
        if prompt == 'eval':
            for i in range(num_devices):
                start_new_thread(start_evaluation, (end_devices[i], test_matrix[:, i]))
        if prompt == 'next':
        """

        server_socket = socket.socket()
        try:
            server_socket.bind(('', 12122))
        except socket.error as e:
            print(str(e))
            exit(1)

        server_socket.listen(1)
        print('Ready to connect')
        client, address = server_socket.accept()
        print('Connected to: ' + address[0] + ':' + str(address[1]))
        data = client.recv(2640)
        print('Data recv')
        if not data:
            break
        print('Cont')
        client.close()
        i += 1
        start_launches(num_devices, output_sizes[i], end_devices, train_matrix, test_matrix, train_true,
                       test_true, epochs)

    while True:
        for output_size in output_sizes:
            prompt = input('"exit"/"eval": ')
            if prompt == 'exit':
                exit(-1)
            if prompt == 'eval':
                for end_device in end_devices:
                    if end_device.output_size == output_size:
                        start_new_thread(start_evaluation, (end_device, test_matrix[:, int(end_device.station_id)]))


if __name__ == '__main__':
    main(100, 16)