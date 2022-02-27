import multiprocessing
import socket
import json
import torch
from torch.autograd import Variable
from network_test.models import CloudModule


def train(model, train_matrix, train_true, optimizer, loss_func):
    model.train()
    if torch.cuda.is_available():
        train_matrix, train_true = train_matrix.cuda(), train_true.cuda()
    data, target = Variable(train_matrix), Variable(train_true)

    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_func(predictions, target)
    loss.backward()
    optimizer.step()

    return predictions


def threaded_client(connection, return_dict):
    print('Worker runned')
    while True:
        data = connection.recv(204800)
        if not data:
            break
        decoded = data.decode()
        data = json.loads(decoded)
        #print(data)
        #data = [json.loads(line) for line in open('data.json', 'r')]
        station_id = data['station_id']
        return_dict[station_id] = data['train']
        #reply = 'Server Says: ' + data.decode('utf-8')
        #connection.sendall(str.encode(reply))
    connection.close()


def main(num_devices, true_value):
    server_socket = socket.socket()
    host = ''
    port = 10203
    thread_count = 0
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        print(str(e))

    model = CloudModule(num_devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    print('Waiting for a Connection..')

    server_socket.listen(num_devices)
    while thread_count < num_devices:
        client, address = server_socket.accept()
        #print('Connected to: ' + address[0] + ':' + str(address[1]))
        #start_new_thread(threaded_client, (client, ))
        p = multiprocessing.Process(target=threaded_client, args=(client, return_dict))
        jobs.append(p)
        p.start()
        thread_count += 1
        print('Thread Number: ' + str(thread_count))
    server_socket.close()

    for proc in jobs:
        proc.join()

    train_data = [[]] * num_devices
    for i in range(num_devices):
        train_data[i] = return_dict[str(i)]

    tensor_matrix = torch.tensor(train_data)
    for i in range(0, tensor_matrix.shape[1]):
        matrix = tensor_matrix[:, i, :]
        train(model, matrix, true_value, optimizer, loss_func)

    # Can do one last prediction


if __name__ == '__main__':
    true = torch.tensor([[1.0], [2.0]])
    num_devices = 2
    main(num_devices, true)
