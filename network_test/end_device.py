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
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        #    with record_function("model_train"):
        test_true = convert_tensor(test_true)
        test_matrix = convert_tensor(test_matrix)
        train_matrix = convert_tensor(train_matrix)
        train_true = convert_tensor(train_true)

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
        """
        self.model.eval()
        output, predictions, ee_p = self.model(test_matrix)

        loss_result = self.loss_func(predictions, test_true)
        r2_loss_result = r2_loss(predictions, test_true)
        ee = early_exit(predictions, test_true, self.ee_range)
        acc, count = binary_acc(ee_p, ee)
        print("Eval loss:", loss_result)
        print("Eval R2:", r2_loss_result)
        print("Binary Acc:", acc)

        if file is not None:
            cuda_time = int(sum([e.self_cuda_time_total for e in prof.profiler.function_events]) / 1000)
            cuda_memory = math.floor(sum([e.cuda_memory_usage for e in prof.profiler.function_events]) / 1e+7) / 100
            cpu_memory = math.floor(sum([e.cpu_memory_usage for e in prof.profiler.function_events]) / 1e+4) / 100
            file.write("{},{},{},{},{},{},{},{},{},{}\n".format(loss1, loss2, loss_result, r2_loss_result, acc, count, int(prof.profiler.self_cpu_time_total / 1000), cuda_time, cpu_memory, cuda_memory))
        """
        if not self.is_offline:
            self.send_data(data_dict)
        print('Ended: ', self.station_id)


if __name__ == "__main__":
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset(history=6)
    #file = open("data/end_device_lr_test", "a", encoding="utf-8")
    start_time = time.time()
    """
    lrs = [0.01, 0.005, 0.001, 0.0005]
    for lr in lrs:
        file.write('2epoch lr' + str(lr) + '\n')
        for i in range(train_matrix.size(dim=1)):
            train_ma = train_matrix[:, i]
            test_ma = test_matrix[:, i]
            ed = EndDevice('0', 16, '127.0.0.1', 10203, is_offline=True)
            ed.create(100, train_ma, train_true, test_ma, test_true, file=file, lr=lr)
        for i in range(train_matrix.size(dim=1)):
    """
    train_ma = train_matrix[:, 0]
    test_ma = test_matrix[:, 0]
    ed = EndDevice('0', 16, '127.0.0.1', 10203, is_offline=True)
    ed.create(100, train_ma, train_true, test_ma, test_true)
    print('Time: ', time.time() - start_time)
