import matplotlib.pyplot as plt
from network_test.pollution_dataset import get_dataset


def main(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        read_list = f.read()
    read_list = read_list.replace('[', '')
    read_list = read_list.replace(']', '')
    read_list = read_list.replace(' ', '')
    read_list = read_list.split(',')
    for index, element in enumerate(read_list):
        read_list[index] = float(element)

    data = get_dataset()
    true_values = []

    for true_value in data[6][1000:1100]:
        true_values.append(true_value[0])

    plt.plot(true_values, label="True")
    plt.plot(read_list[1000:1100], label="Predicted")
    plt.grid(True)
    plt.title("Prediction Compared to True Value")
    plt.xlabel("Measurements Ordered by Date")
    plt.ylabel("μg/m^3")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main("../network_test/data/result_graph")
