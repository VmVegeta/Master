from network_test.models import RnnLayer, NnLayer
from network_test.pollution_dataset import get_dataset
import time
from network_test.client_base import *
from torch.profiler import profile, record_function, ProfilerActivity


class DeviceModel(nn.Module):
    def __init__(self, output_size: int, rnn_size: int):
        super(DeviceModel, self).__init__()
        self.model = nn.Sequential(
            RnnLayer(1, rnn_size),
            NnLayer(rnn_size, 64),
            NnLayer(64, 64),
            NnLayer(64, output_size)
        )
        self.regression = nn.Linear(output_size, 1)
        self.early_exit_classification = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.Linear(32, 1))

    def forward(self, x):
        h = self.model(x)
        return h, self.regression(h), self.early_exit_classification(h)


class EndDevice(ClientBase):
    def __init__(self, station_id: str, output_size: int, server_address: str, server_port: int, is_offline=False, rnn_size=16, ee_range=8):
        model = DeviceModel(output_size, rnn_size)
        super().__init__(model, station_id, output_size, server_address, server_port, ee_range)
        #torch.manual_seed(2022)
        self.is_offline = is_offline

        #self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.LSTM}, dtype=torch.float16) #Don't work with CUDA
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss_func = nn.MSELoss()
        self.binary_loss = nn.BCEWithLogitsLoss()

    def only_early_exit_train(self, X, y, optimizer, last):
        self.model.train()
        optimizer.zero_grad()
        output, predictions, to_exit_prediction = self.model(X)
        to_exit = early_exit(predictions, y, self.ee_range)
        exit_loss = self.binary_loss(to_exit_prediction, to_exit)
        exit_loss.backward()
        optimizer.step()
        loss = None
        if last:
            print('{:.4f}'.format(r2_loss(predictions, y)))
            loss = self.loss_func(predictions, y)
            print('{:.4f}'.format(loss))
        return output, loss

    def create(self, epochs, train_matrix, train_true, test_matrix, test_true, file=None, lr=0.008):
        print('Started: ', self.station_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)

        data_dict = {"station_id": self.station_id, 'train': []}
        test_matrix = convert_tensor(test_matrix)
        train_matrix = convert_tensor(train_matrix)
        train_true = convert_tensor(train_true)
        test_true = convert_tensor(test_true)

        for epoch in range(1, int(epochs)):
            to_print = epoch == epochs - 1 and self.is_offline
            output, loss = self.train(train_matrix, train_true, optimizer, to_print)

        if True:
            epochs = int(epochs * 2)
            for epoch in range(1, epochs):
                to_send = epochs - 1 == epoch
                to_print = to_send and self.is_offline
                output, loss2 = self.early_exit_train(train_matrix, train_true, optimizer, to_print)
                if to_send:
                    data_dict['train'].append(quantize_data(output))
        if file is not None or True:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                with record_function("model_train"):
                    self.model.eval()
                    output, predictions, ee_p = self.model(test_matrix)

                    loss_result = self.loss_func(predictions, test_true)
                    r2_loss_result = r2_loss(predictions, test_true)
                    pred_ee = get_early_exit(ee_p)
            ee = early_exit(predictions, test_true, self.ee_range)
            acc, count = binary_accuracy(ee_p, ee)
            print("Eval loss:", loss_result)
            print("Eval R2:", r2_loss_result)
            print("Binary Acc:", acc)

            cuda_time = sum([event.self_cuda_time_total for event in prof.profiler.function_events])
            cpu_memory = sum([event.cpu_memory_usage for event in prof.profiler.function_events])
            cuda_memory = sum([event.cuda_memory_usage for event in prof.profiler.function_events])
            #file.write("{},{},{},{},{},{},{},{},{},{}\n".format(loss, loss2, loss_result, r2_loss_result, acc, count, prof.profiler.self_cpu_time_total, cuda_time, cpu_memory, cuda_memory))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(cuda_time)

        if not self.is_offline:
            self.send_data(data_dict)
        print('Ended: ', self.station_id)


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, sn, t, d = get_dataset(history=6)
    file = open("data/end_device_perf_w", "a", encoding="utf-8")
    start_time = time.time()

    """
    for _ in range(10):
        for i in range(train_matrix.size(dim=1)):
            train_ma = train_matrix[:, i]
            test_ma = test_matrix[:, i]
            ed = EndDevice('0', 8, '127.0.0.1', 10203, is_offline=True)
            ed.create(100, train_ma, train_true, test_ma, test_true, file=file)
    """
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    ed = EndDevice('0', 8, '127.0.0.1', 10203, is_offline=True)
    ed.create(100, train_ma, train_true, test_ma, test_true)
    print('Time: ', time.time() - start_time)
