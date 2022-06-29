from __future__ import print_function

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

import network_test.pollution_dataset as pd
import rnn_net
cudnn.benchmark = True


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def train(model, train_matrix, train_true, optimizer, num_devices, loss_func, is_last, file, train_input):
    model.train()
    model_losses = [0] * (num_devices + 1 + 3) # Added 3 for middle
    #for batch_idx, (data, target) in enumerate(train_loader):
    if torch.cuda.is_available():
        train_matrix, train_true, train_input = train_matrix.cuda(), train_true.cuda(), train_input.cuda()
    data, target = Variable(train_matrix), Variable(train_true)
    optimizer.zero_grad()
    predictions = model(data, train_input)
    total_loss = 0

    # for each prediction (num_devices + 1 for cloud), add to loss
    for i, prediction in enumerate(predictions):
        loss = loss_func(prediction, target)
        model_losses[i] += loss.sum()
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    if is_last:
        file.write("{:.4f},{:.4f},".format(model_losses[-1], r2_loss(predictions[-1], target)))

    """
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss.item())
                        for i, loss in enumerate(model_losses[:-1])])
    print('Train Loss:: {}, cloud: {:.4f}'.format(loss_str, model_losses[-1].item()))
    """
    return model_losses


def train_model(model, model_path, data, lr, epochs, num_devices):
    train_matrix, train_true, test_matrix, test_true, station_names, ordered_m, ordered_t, train_input, test_input = data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    file = open("corrected_memory", 'a', encoding="utf-8")

    for epoch in range(1, epochs):
        is_last = epoch == epochs - 1
        train(model, train_matrix, train_true, optimizer, num_devices, loss_func, is_last, file, train_input)
        #torch.save(model, model_path)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("model_inference"):
            run_test_set(model, test_matrix, test_true, num_devices, loss_func, file, test_input)

    cuda_time = sum([e.self_cuda_time_total for e in prof.profiler.function_events])
    cpu_memory = sum([e.self_cpu_memory_usage if e.self_cpu_memory_usage > 0 else 0 for e in prof.profiler.function_events])
    cuda_memory = sum([e.self_cuda_memory_usage if e.self_cuda_memory_usage > 0 else 0 for e in prof.profiler.function_events])
    file.write('{},{},{},{}\n'.format(prof.profiler.self_cpu_time_total, cuda_time, cpu_memory, cuda_memory))
    file.close()


def run_test_set(model, test_matrix, test_true, num_devices, loss_func, file, test_input):
    model.eval()
    model_losses = [0] * (num_devices + 1 + 3)
    num_correct = [0] * (num_devices + 1 + 3)
    r2_list = [0] * (num_devices + 1 + 3)

    # for data, target in tqdm(test_loader, leave=False):
    if torch.cuda.is_available():
        test_matrix, test_true, test_input = test_matrix.cuda(), test_true.cuda(), test_input.cuda()
    data, target = Variable(test_matrix), Variable(test_true)
    predictions = model(data, test_input)

    for i, prediction in enumerate(predictions):
        loss = loss_func(prediction, target)
        model_losses[i] += loss.sum()
        r2_list[i] += r2_loss(prediction, target)

    loss_str = ', '.join(['{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['{}: {:.4f}'.format(i, r2)
                        for i, r2 in enumerate(r2_list[:-1])])
    print('Test  Loss:: {}, cloud: {:.4f}'.format(loss_str, model_losses[-1]))
    print('Test  Acc.:: {}, cloud: {:.4f}'.format(acc_str, r2_list[-1]))
    file.write("{:.4f},{:.4f},".format(model_losses[-1], r2_list[-1]))
    return model_losses, num_correct


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output', default='models/model.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    """
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    """
    data = pd.get_dataset(history=6)
    for i in range(10):
        train_matrix = data[0]
        num_devices = data[0].shape[1]
        in_channels = train_matrix.shape[len(train_matrix.shape) - 1]
        model = rnn_net.DDNN(in_channels, num_devices)
        if args.cuda:
            model = model.cuda()
        train_model(model, args.output, data, args.lr, args.epochs, num_devices)
