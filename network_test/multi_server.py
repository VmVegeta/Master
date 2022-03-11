import multiprocessing
import socket
import json
import torch
from torch.autograd import Variable
from network_test.models import CloudModule
from network_test.pollution_dataset import get_dataset


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def train(model, train_matrix, train_true, optimizer, loss_func, current_epoch):
    model.train()
    if torch.cuda.is_available():
        train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
    data, target = Variable(train_matrix), Variable(train_true)

    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_func(predictions, target)
    loss.backward()
    optimizer.step()

    if current_epoch % 5 == 0 or current_epoch > 189:
        print(loss)
        print(r2_loss(predictions, target))

    return predictions


def predict(model, matrix):
    model.eval()
    if torch.cuda.is_available():
        matrix = matrix.cuda()
    data = Variable(matrix)

    return model(data)


def train_client_thread(connection, child_pipe):
    json_data = ''
    while True:
        data = connection.recv(26480000)
        if not data:
            break
        json_data += data.decode()
    data = json.loads(json_data)
    connection.close()

    station_id = data['station_id']
    child_pipe.send((station_id, data['train']))
    print('Thread done')


def infer_client_thread(connection, child_pipe):
    json_data = ''
    while True:
        data = connection.recv(26480000)
        if not data:
            break
        json_data += data.decode()
    data = json.loads(json_data)
    connection.close()

    station_id = data['station_id']
    child_pipe.send((station_id, data['infer'], data['datetime']))


def wait_and_train(server_socket, model, optimizer, loss_func, true_value, num_devices):
    thread_count = 0
    jobs = []
    parent_pipes = []
    while thread_count < num_devices:
        client, address = server_socket.accept()
        print('Connected to: ' + address[0] + ':' + str(address[1]))
        # start_new_thread(threaded_client, (client, ))
        #ret_value = multiprocessing.Value("d", 0.0, lock=False)
        parent_pipe, child_pipe = multiprocessing.Pipe()
        parent_pipes.append(parent_pipe)
        p = multiprocessing.Process(target=train_client_thread, args=(client, child_pipe))
        jobs.append(p)
        p.start()
        thread_count += 1
        print('Thread Number: ' + str(thread_count))

    train_data = [[]] * num_devices
    for parent_pipe in parent_pipes:
        received = parent_pipe.recv()
        train_data[int(received[0])] = received[1]
        print("Works: ", received[0])

    for proc in jobs:
        proc.join()

    print("Workers done")

    tensor_matrix = torch.tensor(train_data)
    tensor_matrix = torch.dequantize(tensor_matrix)
    #for i in range(tensor_matrix.shape[1]):
    #matrix = tensor_matrix[:, i, :]
    for i in range(200):
        matrix = tensor_matrix[:, i % 10, :]
        train(model, matrix, true_value, optimizer, loss_func, i)


def print_results(predictions, true_value, loss_func):
    if torch.cuda.is_available():
        true_value = true_value.cuda()
    true_value = Variable(true_value)

    loss = loss_func(predictions, true_value)
    print(loss)
    print(r2_loss(predictions, true_value))


def start_eval(server_socket, model, true_value):
    print('Eval Ready')
    thread_count = 0
    jobs = []
    parent_pipes = []
    while thread_count < num_devices:
        client, address = server_socket.accept()
        print('Connected to: ' + address[0] + ':' + str(address[1]))
        parent_pipe, child_pipe = multiprocessing.Pipe()
        parent_pipes.append(parent_pipe)
        p = multiprocessing.Process(target=train_client_thread, args=(client, child_pipe))
        jobs.append(p)
        p.start()
        thread_count += 1
        print('Thread Number: ' + str(thread_count))

    train_data = [[]] * num_devices
    for parent_pipe in parent_pipes:
        received = parent_pipe.recv()
        train_data[int(received[0])] = received[1]

        print("Works: ", received[0])

    for proc in jobs:
        proc.join()

    tensor_matrix = torch.tensor(train_data)
    tensor_matrix = torch.dequantize(tensor_matrix)
    predictions = predict(model, tensor_matrix)
    print_results(predictions, true_value, torch.nn.MSELoss())


def start_inference(server_socket, model):
    print('Inference Ready')
    while True:
        thread_count = 0
        jobs = []
        parent_pipes = []
        while thread_count < num_devices:
            client, address = server_socket.accept()
            print('Connected to: ' + address[0] + ':' + str(address[1]))
            parent_pipe, child_pipe = multiprocessing.Pipe()
            parent_pipes.append(parent_pipe)
            p = multiprocessing.Process(target=infer_client_thread, args=(client, child_pipe))
            jobs.append(p)
            p.start()
            thread_count += 1
            print('Thread Number: ' + str(thread_count))

        datetime = None
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

        tensor_matrix = torch.tensor(train_data)
        tensor_matrix = torch.dequantize(tensor_matrix)
        prediction = predict(model, tensor_matrix)
        print('Predicted value for Bygdø Alle ', datetime, '{:.4f}'.format(prediction))


def main(num_devices, true_value, test_true):
    server_socket = socket.socket()
    host = ''
    port = 10203
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        print(str(e))
        exit(1)

    model = CloudModule(num_devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    #return_dict = {}
    #queue = multiprocessing.Queue()
    #queue.put(return_dict)

    print('Ready to connect')

    server_socket.listen(num_devices)
    wait_and_train(server_socket, model, optimizer, loss_func, true_value, num_devices)
    #start_new_thread(wait_and_train, (server_socket, return_dict, model, optimizer, loss_func, true_value, num_devices))

    start_eval(server_socket, model, test_true)

    server_socket.close()
    print("Socket Close")
    #return model


if __name__ == '__main__':
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = 7
    main(num_devices, train_true, test_true)
