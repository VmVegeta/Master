import network_test.end_device as end_device
import network_test.multi_server as multi_server
from _thread import *
from network_test.pollution_dataset import get_dataset
import torch


def launch_end_device(device_id: int, epochs: int, train_matrix: torch.Tensor, train_ture: torch.Tensor, test_matrix, test_true):
    end_device.create(device_id, train_matrix, train_ture, '127.0.0.1', 10203, epochs, test_matrix, test_true)


def launch_cloud_device(num_devices, train_true):
    multi_server.main(num_devices, train_true)


def main(epochs: int):
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = train_matrix.shape[1]
    #launch_cloud_device(num_devices, train_true)
    for device_id in range(num_devices):
        device_train_matrix = train_matrix[:, device_id]
        device_test_matrix = test_matrix[:, device_id]
        start_new_thread(launch_end_device, (device_id, epochs, device_train_matrix, train_true, device_test_matrix, test_true))

    while True:
        prompt = input('"exit" to exit:')
        if prompt == 'exit':
            break


if __name__ == '__main__':
    main(100)