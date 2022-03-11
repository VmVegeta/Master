from network_test.end_device import EndDevice
from _thread import start_new_thread
from network_test.pollution_dataset import get_dataset
import torch


def launch_end_device(end_device: EndDevice, epochs: int, train_matrix: torch.Tensor, train_ture: torch.Tensor, test_matrix, test_true):
    end_device.create(epochs, train_matrix, train_ture, test_matrix, test_true)


def start_evaluation(end_device: EndDevice, test_matrix: torch.Tensor):
    end_device.evaluate(test_matrix)


def main(epochs: int):
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = train_matrix.shape[1]

    end_devices = []
    for device_id in range(num_devices):
        end_device = EndDevice(str(device_id), '127.0.0.1', 10203)
        end_devices.append(end_device)
        device_train_matrix = train_matrix[:, device_id]
        device_test_matrix = test_matrix[:, device_id]

        start_new_thread(
            launch_end_device,
            (end_device, epochs, device_train_matrix, train_true, device_test_matrix, test_true)
        )

    while True:
        prompt = input('"exit"/"eval"')
        if prompt == 'exit':
            exit(-1)
        if prompt == 'eval':
            for i in range(num_devices):
                start_new_thread(start_evaluation, (end_devices[i], test_matrix[:, i]))


if __name__ == '__main__':
    main(100)