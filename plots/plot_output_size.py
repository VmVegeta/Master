from typing import Dict, List
import matplotlib.pyplot as plt


def get_values(filename):
    data = {}
    key = ""
    count = 0
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            rest = count % 5
            if rest == 0:
                key = line
            elif rest == 1:
                handle_default(data, key + '-data')
                data[key + '-data'].append(int(line))
            elif rest == 2:
                accuracies = line.split(',')
                handle_default(data, key + '-loss')
                data[key + '-loss'].append(float(accuracies[0]))
                handle_default(data, key + '-r2')
                data[key + '-r2'].append(float(accuracies[1]))
            elif rest == 3:
                handle_default(data, key + '-data-eval')
                data[key + '-data-eval'].append(int(line))
            else:
                accuracies = line.split(',')
                handle_default(data, key + '-loss-eval')
                data[key + '-loss-eval'].append(float(accuracies[0]))
                handle_default(data, key + '-r2-eval')
                data[key + '-r2-eval'].append(float(accuracies[1]))
            count += 1
    return data


def plot(data: Dict[str, List]):
    output_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    losses = []
    for output_size in output_sizes:
        total = data[str(output_size) + '-loss'] + data[str(output_size) + '-loss-eval']
        sum_total = sum(total)
        len_total = len(total)
        losses.append(sum_total / len_total)

    plt.plot(output_sizes, losses)
    plt.title("Accuracy Based on Output Size")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Output Size")
    plt.axis([0, 66, 0, 600])
    plt.grid(True)
    plt.show()
    """
    r2s = []
    for output_size in output_sizes:
        total = data[str(output_size) + '-r2'] + data[str(output_size) + '-r2-eval']
        sum_total = sum(total)
        len_total = len(total)
        r2 = sum_total / len_total
        r2s.append(r2)
    plt.plot(output_sizes, r2s)
    plt.title("Accuracy Divided on Amount of Data Sent")
    plt.ylabel("R2-Score / Single Station Sent Byte (Eval-Set)")
    plt.xlabel("Output Size")
    plt.xlim((0, 66))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()
    
    """
    r2s = []
    for output_size in output_sizes:
        total = data[str(output_size) + '-r2'] + data[str(output_size) + '-r2-eval']
        sum_total = sum(total)
        len_total = len(total)
        r2 = sum_total / len_total

        total = data[str(output_size) + '-data-eval']
        sum_total = sum(total)
        len_total = len(total)
        data_size = sum_total / len_total / 3
        r2s.append(data_size)

    plt.plot(output_sizes, r2s)
    plt.title("Accuracy Divided on Amount of Data Sent")
    plt.ylabel("R2-Score / Single Station Sent Byte (Eval-Set)")
    plt.xlabel("Output Size")
    plt.xlim((0, 66))
    plt.grid(True)
    plt.show()


def plot_two(data: Dict[str, List], nq_data: Dict[str, List]):
    output_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    losses = []
    nq_losses = []
    for output_size in output_sizes:
        total = data[str(output_size) + '-loss'] + data[str(output_size) + '-loss-eval']
        sum_total = sum(total)
        len_total = len(total)
        losses.append(sum_total / len_total)

        total = nq_data[str(output_size) + '-loss'] + nq_data[str(output_size) + '-loss-eval']
        sum_total = sum(total)
        len_total = len(total)
        nq_losses.append(sum_total / len_total)

    plt.plot(output_sizes, losses, label="Quantitation")
    plt.plot(output_sizes, nq_losses, label="Float")
    plt.title("Accuracy Based on Output Size")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Output Size")
    plt.axis([0, 66, 0, 600])
    plt.grid(True)
    plt.legend()
    plt.show()

    r2s = []
    nq_r2s = []
    for output_size in output_sizes:
        total = data[str(output_size) + '-r2'] + data[str(output_size) + '-r2-eval']
        sum_total = sum(total)
        len_total = len(total)
        r2 = sum_total / len_total

        total = data[str(output_size) + '-data-eval']
        sum_total = sum(total)
        len_total = len(total)
        data_size = sum_total / len_total / 3
        r2s.append(r2 / data_size)

        total = nq_data[str(output_size) + '-r2'] + nq_data[str(output_size) + '-r2-eval']
        sum_total = sum(total)
        len_total = len(total)
        r2 = sum_total / len_total

        total = nq_data[str(output_size) + '-data-eval']
        sum_total = sum(total)
        len_total = len(total)
        data_size = sum_total / len_total / 3
        nq_r2s.append(r2 / data_size)

    plt.plot(output_sizes, r2s, label="Quantitation")
    plt.plot(output_sizes, nq_r2s, label="Float")
    plt.title("Accuracy Divided on Amount of Data Sent")
    plt.ylabel("R2-Score / Single Station Sent Byte (Eval-Set)")
    plt.xlabel("Output Size")
    plt.xlim((0, 66))
    plt.grid(True)
    plt.legend()
    plt.show()


def print_avg(data: Dict[str, List]):
    for key, values in data.items():
        length = len(values)
        total = sum(values)
        print(key, str(total/length))


def handle_default(data, key):
    value = data.get(key)
    if value is None:
        data[key] = []


if __name__ == "__main__":
    nq_data = get_values("../network_test/data/server_side_not_quant")
    data = get_values("../network_test/data/server_side")
    #plot(nq_data)
    print_avg(data)
