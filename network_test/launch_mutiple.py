from network_test.end_device import EndDevice
from _thread import start_new_thread
from network_test.pollution_dataset import get_dataset
import torch


def launch_end_device(end_device: EndDevice, epochs: int, train_matrix: torch.Tensor, train_ture: torch.Tensor, test_matrix, test_true):
    end_device.create(epochs, train_matrix, train_ture, test_matrix, test_true)


def main(epochs: int):
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = train_matrix.shape[1]

    clients = []
    for device_id in range(num_devices):
        end_device = EndDevice(device_id, '127.0.0.1', 10203)
        clients.append(end_device)
        device_train_matrix = train_matrix[:, device_id]
        device_test_matrix = test_matrix[:, device_id]

        start_new_thread(
            launch_end_device,
            (end_device, epochs, device_train_matrix, train_true, device_test_matrix, test_true)
        )

    while True:
        prompt = input('"exit"/"infer"')
        if prompt == 'exit':
            exit(-1)
        if prompt == 'infer':
            break


if __name__ == '__main__':
    main(20)