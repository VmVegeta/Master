import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate
import numpy as np


def get_values(filename, name):
    columns = ['Type', 'MSE-Loss', 'R2', 'MSE', 'Eval-R2', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']
    #columns = ['MSE-Loss', 'R2', 'MSE-Loss-Eval', 'Eval-R2', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']
    data = pd.DataFrame(columns=columns)
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            values = line.split(',')
            for i in range(4):
                values[i] = float(values[i])
            for i in range(4, 8):
                values[i] = int(values[i])

            values = [[name] + values]
            #values = [values]

            test = pd.DataFrame(values, columns=columns)
            data = pd.concat([data, test])
    return data


def get_middle_values(filename, name):
    columns = ['MSE-Loss', 'MSE-Loss-Eval', 'Eval-R2', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']
    columns = ['MSE-Loss', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']
    data = pd.DataFrame(columns=columns)
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            values = line.split(',')
            for i in range(1):
                values[i] = float(values[i])
            for i in range(1, 5):
                values[i] = int(values[i])

            values = [values]

            test = pd.DataFrame(values, columns=columns)
            data = pd.concat([data, test])
    return data


def get_end_values(filename, name):
    columns = ['Type', 'Loss1', 'R2-1', 'MSE-Loss', 'R2', 'EE-Acc',
               'Count', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']
    columns = ['Loss1', 'R2-1', 'MSE-Loss', 'R2', 'EE-Acc',
               'Count', 'CPU-Time', 'CUDA-Time', 'CPU-Memory', 'CUDA-Memory']

    data = pd.DataFrame(columns=columns)
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            values = line.split(',')
            for i in range(6):
                values[i] = float(values[i])

            if values[6] == 'nan':
                values[6] = 0
            else:
                values[6] = int(values[6])

            for i in range(7, 10):
                values[i] = int(values[i])

            #values = [[name] + values]
            values = [values]

            test = pd.DataFrame(values, columns=columns)
            data = pd.concat([data, test])
    return data


def show_data(df_list):
    data = pd.concat(df_list)
    sns.boxplot(x="Type", y="MSE", data=data)
    plt.ylim((0, 135))
    plt.title("Box-Plot Distributed and Centralized Train-Set MSE")
    plt.grid(True, axis='y')
    plt.show()


def show_eval_data(df_list):
    data = pd.concat(df_list)
    sns.boxplot(x="Type", y="MSE", data=data)
    plt.ylim((0, 250))
    plt.title("Box-Plot Distributed and Centralized Eval-Set MSE")
    plt.grid(True, axis='y')
    plt.show()


def describe(data_frame):
    desc = data_frame.astype(float).describe()
    print(tabulate(desc, headers='keys', tablefmt='psql'))
    #display(desc)
    #print(desc)


def averages(server, middle, end, column_name):
    server_avg = server[column_name].mean()
    middle_avg = middle[column_name].mean() * 3
    end_avg = end[column_name].mean() * 7
    print(str(server_avg), str(middle_avg), str(end_avg))
    sum = server_avg + middle_avg + end_avg
    print(str(sum))
    return sum


def create_stacked_bar_graph(center, server, middle, end, column_name):
    middle_mean = middle[column_name].mean() - 9393
    end_mean = end[column_name].mean() - 9393
    cent_avg = center[column_name].mean() / 1000
    server_avg = [0, server[column_name].mean() / 1000]
    middle_avg = [0, middle_mean * 3 / 1000]
    bellow_end_avg = [0, (middle_mean * 3 + server[column_name].mean()) / 1000]
    end_avg = [0, end_mean * 7 / 1000]
    labels = ['Centralized', 'Distributed']

    fig, ax = plt.subplots()

    ax.bar(labels, cent_avg, edgecolor='black')
    ax.bar(labels, server_avg, label='Server', edgecolor='black')
    ax.bar(labels, middle_avg, bottom=server_avg, label='Middle', edgecolor='black')
    ax.bar(labels, end_avg, bottom=bellow_end_avg, label='End', edgecolor='black')

    ax.legend()
    plt.title("Centralized and Distributed Time Used CUDA Whole Eval-Set No Tensor.tolist()")
    plt.ylabel("ms")
    plt.xlabel("Type")
    plt.show()


def create_bar_graph(center, server, middle, end, column_name):
    middle_mean = middle[column_name].mean() / 1000# - 9393
    end_mean = end[column_name].mean() / 1000# - 9393
    cent_avg = center[column_name].mean() / 1000
    server_avg = server[column_name].mean() / 1000
    bar_data = [cent_avg, server_avg, middle_mean, end_mean]
    labels = ['Centralized', 'Server', 'Middle', 'End']

    fig, ax = plt.subplots()

    ax.bar(labels, [cent_avg, 0, 0, 0], edgecolor='black')
    ax.bar(labels, [0, server_avg, 0, 0], edgecolor='black')
    ax.bar(labels, [0, 0, middle_mean,  0], edgecolor='black')
    ax.bar(labels, [0, 0, 0, end_mean], edgecolor='black')

    plt.title("Mean CUDA-Time Used By Single Step/Centralized Server")
    plt.ylabel("ms")
    plt.xlabel("Units")
    plt.show()


if __name__ == '__main__':
    center_d = get_values("../centerlized/centralized_data_default", "Default")
    center = get_values("../centerlized/centralized_data", "Center")
    center_pr8 = get_values("../centerlized/prev_8", "Prev 8")
    center_pr = get_values("../centerlized/centralized_data_prev12", "Prev 12")
    center_pr16 = get_values("../centerlized/prev_16", "Prev 16")
    center_pr24 = get_values("../centerlized/centralized_data_prev24", "Prev 24")

    df_list = [center, center_d, center_pr8, center_pr, center_pr16, center_pr24]
    show_eval_data(df_list)
    """
    server_side = get_values("../network_test/data/server_side_perf", "Cloud")
    middle = get_middle_values("../network_test/data/middle_pref2.txt", "Middle")
    end = get_end_values("../network_test/data/end_device_perf_w", "End")
    end_wo = get_end_values("../network_test/data/end_device_perf", "End")
    
    
    #describe(server_side)
    #describe(middle)
    describe(end)
    describe(end_wo)
    #describe(center)
    name = 'CUDA-Time'
    averages(server_side, middle, end, name)
    create_bar_graph(center, server_side, middle, end, name)
    print(center[name].mean())
    #show_eval_data(server_side, center)
    """
