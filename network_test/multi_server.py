import multiprocessing
import socket
import json
import torch
from torch.autograd import Variable
from network_test.models import CloudModule
from _thread import *
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


def threaded_client(connection, child_pipe):
    datas = ''
    while True:
        data = connection.recv(26480000)
        if not data:
            break
        datas += data.decode()
    data = json.loads(datas)
    connection.close()

    #return_dict = queue.get()
    station_id = data['station_id']
    #return_dict[station_id] = data['train']
    #queue.put(return_dict)
    child_pipe.send((station_id, data['train']))
    print('Thread done')


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
        p = multiprocessing.Process(target=threaded_client, args=(client, child_pipe))
        jobs.append(p)
        p.start()
        thread_count += 1
        print('Thread Number: ' + str(thread_count))
    server_socket.close()
    print("Socket Close")

    train_data = [[]] * num_devices
    for parent_pipe in parent_pipes:
        received = parent_pipe.recv()
        train_data[int(received[0])] = received[1]
        print("Works: ", received[0])

    for proc in jobs:
        proc.join()

    print("Workers done")

    tensor_matrix = torch.tensor(train_data)
    #for i in range(tensor_matrix.shape[1]):
    #matrix = tensor_matrix[:, i, :]
    for i in range(200):
        matrix = tensor_matrix[:, i % 10, :]
        train(model, matrix, true_value, optimizer, loss_func, i)


def main(num_devices, true_value):
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

    print('Waiting for a Connection..')

    server_socket.listen(num_devices)
    wait_and_train(server_socket, model, optimizer, loss_func, true_value, num_devices)
    #start_new_thread(wait_and_train, (server_socket, return_dict, model, optimizer, loss_func, true_value, num_devices))

    # Can do one last prediction
    #return model


if __name__ == '__main__':
    train_matrix, train_true, test_matrix, test_true, station_names = get_dataset()
    num_devices = 7
    #true = torch.tensor([[1.0], [2.0]])
    main(num_devices, train_true)
