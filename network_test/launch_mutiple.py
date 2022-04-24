from network_test.end_device import EndDevice
from network_test.middle_step import MiddleDevice
from _thread import start_new_thread
from network_test.pollution_dataset import get_dataset
import torch
import socket


def launch_end_device(end_device: EndDevice, epochs: int, train_matrix: torch.Tensor, train_true: torch.Tensor,
                      test_matrix, test_true):
    end_device.create(epochs, train_matrix, train_true, test_matrix, test_true)


def start_evaluation(end_device: EndDevice, test_matrix: torch.Tensor):
    end_device.evaluate(test_matrix)


def launch_middle(middle_device: MiddleDevice, train_true: torch.Tensor, test_true: torch.Tensor):
    middle_device.create(train_true, test_true)


def start_launches(num_devices, output_size, end_devices, train_matrix, test_matrix, train_true, test_true, epochs):
    ports = [11203, 11203, 12203, 12203, 13203, 13203, 13203]
    device_ids = ['0', '1', '0', '1', '0', '1', '2']
    for device_id in range(num_devices):
        end_device = EndDevice(device_ids[device_id], output_size, '127.0.0.1', ports[device_id])
        #end_device = EndDevice(str(device_id), output_size, '127.0.0.1', 10203)
        end_devices.append(end_device)
        device_train_matrix = train_matrix[:, device_id]
        device_test_matrix = test_matrix[:, device_id]

        start_new_thread(
            launch_end_device,
            (end_device, epochs, device_train_matrix, train_true, device_test_matrix, test_true)
        )


def start_middle_launches(num_devices, output_size, train_true, test_true):
    ports = [11203, 12203, 13203]
    devices = [2, 2, 3]
    for device_id in range(num_devices):
        middle_device = MiddleDevice(str(device_id), output_size, devices[device_id], '', '127.0.0.1', ports[device_id], 10203)

        start_new_thread(
            launch_middle,
            (middle_device, train_true, test_true)
        )


def main(epochs: int, output_size: int):
    train_matrix, train_true, test_matrix, test_true, station_names, ordered_matrix, ordered_true_values = get_dataset()
    num_devices = train_matrix.shape[1]
    ordered_matrix = torch.Tensor(ordered_matrix)
    ordered_true_values = torch.Tensor(ordered_true_values)
    """
    end_devices = []
    start_middle_launches(3, 16, train_true, test_true)
    start_launches(num_devices, 16, end_devices, train_matrix, test_matrix, train_true, test_true, epochs)
    """
    output_sizes = [1, 1, 2, 2, 32, 64]
    #              [32, 32, 36, 36, 36, 40, 40, 40, 44, 44, 44, 48, 48, 48, 52, 52, 52, 56, 56, 56, 60, 60, 60, 64, 64, 64]
    i = 0
    while i < len(output_sizes):
        end_devices = []
        start_middle_launches(3, output_sizes[i], train_true, test_true)
        start_launches(num_devices, output_sizes[i], end_devices, train_matrix, test_matrix, train_true, test_true, epochs)

        listen_wait()

        for index in range(num_devices):
            start_new_thread(start_evaluation, (end_devices[index], test_matrix[:, index]))

        listen_wait()
        i += 1

    while True:
        prompt = input('"exit"/"eval": ')
        if prompt == 'exit':
            exit(-1)
        if prompt == 'eval':
            for index, end_device in enumerate(end_devices):
                start_new_thread(start_evaluation, (end_device, test_matrix[:, index]))


def listen_wait():
    server_socket = socket.socket()
    try:
        server_socket.bind(('', 12122))
    except socket.error as e:
        print(str(e))
        exit(1)

    server_socket.listen(1)
    client, address = server_socket.accept()
    data = client.recv(2640)
    if not data:
        exit(-1)
    client.close()


if __name__ == '__main__':
    main(100, 16)
