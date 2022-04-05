from builtins import enumerate

from measure import Measure
from typing import Dict, List
from readers.ReadMeasurments import read_measurements
import torch
import random
from torch.autograd import Variable
import torch.utils.data as data


def get_difference_in_hours(measure: Measure, earlier_measure: Measure):
    """
        Function to return the difference
        in hours between two measurements

        Args:
            measure (Measure): The latest measurement object by time
            earlier_measure (Measure): The earliest measurement
                object by time
    """
    measure_time = measure.datetime
    earlier_time = earlier_measure.datetime
    diff = earlier_time - measure_time
    seconds = diff.total_seconds()
    hours = int(seconds / 60 / 60)
    return hours


def get_station_means(measurements: List[Measure]):
    sum = 0
    for measurement in measurements:
        sum += measurement.value
    return sum / len(measurements)


def split_set(data, train_proportion):
    number_of_elements = len(data)
    train_size = int(number_of_elements * train_proportion)
    train = data[:train_size]
    test = data[train_size:]
    return train, test


def shuffle_data(matrix, true_values):
    random.seed(2022)
    random.shuffle(matrix)
    random.seed(2022)
    random.shuffle(true_values)

    return matrix, true_values


def build_parallel_matrix(result_list: List[Measure], input_data_list: List[List[Measure]], history=6):
    """
        Function to build input matrixes for each station and a list of true values.
        The matrixes contains a measurement for every hour back based the size of history args.
        It will only return valid matrixes, valid matrixes where all hours in history of all
        stations has valid measurements and target station has valid measurement.

        Args:
            result_list (List[Measure]): A list of measurement object for the target station
            input_data_list (List[List[Measure]]): A list of all measurement objects for all
                stations except the target station
            history (int): The number of hours to go back in history for sequence
        """
    length_idl = len(input_data_list)

    # List of indexes to keep current position in stations measurements
    indexes = []
    while len(indexes) < length_idl:
        indexes.append(0)

    matrixes = []
    true_value = []
    for predict_measurement in result_list:
        current_datetime = predict_measurement.datetime
        matrix = []
        for station_index, input_data in enumerate(input_data_list):
            single_input = []
            while True:
                # Gets measurement of index if index is within range of list
                measurement = input_data[indexes[station_index]] if indexes[station_index] < len(input_data) else None
                # Breaks if out of index
                if measurement is None:
                    break

                # Breaks if measurement of station is ahead of current target measurement
                if current_datetime < measurement.datetime:
                    break

                if current_datetime == measurement.datetime:
                    prev = None
                    # Go through all previous index to populate sequence
                    for reduce_index in range(0, history):
                        new_index = indexes[station_index] - reduce_index
                        # Break if history do not far enough back
                        if new_index < 0:
                            break

                        current = input_data[new_index]
                        if prev is not None:
                            # Break if previous measurement is a hour earlier
                            if get_difference_in_hours(current, prev) != 1:
                                break
                        prev = current
                        # Adds measurement value to a sequence
                        single_input.append([current.value])

                    indexes[station_index] += 1
                    break
                indexes[station_index] += 1

            # Add to matrix if success fully added history
            if len(single_input) == history:
                matrix.append(single_input)

        # Add to matrixes and true value if matrix for all stations added
        if len(matrix) == length_idl:
            matrixes.append(matrix)
            true_value.append([predict_measurement.value])

    print(len(true_value))
    return matrixes, true_value


def create_loader(input, true, batch_size):
    train_matrix, train_true = Variable(input), Variable(true)
    torch_dataset = data.TensorDataset(train_matrix, train_true)
    return data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True)


def predict_parallel_all(station_data: Dict[str, List[Measure]], station_name: str, history=6):
    result_data = station_data[station_name]
    input_data_list = []
    station_names = []
    for measurement_station in station_data:
        input_data = station_data[measurement_station]
        if measurement_station == station_name or len(input_data) < 40000:
            continue
        input_data_list.append(input_data)
        station_names.append(measurement_station)
    matrix, true_values = build_parallel_matrix(result_data, input_data_list, history)
    matrix, true_values = shuffle_data(matrix, true_values)

    #TODO: Change this
    train_matrix, test_matrix = split_set(matrix, 0.7)
    train_true, test_true = split_set(true_values, 0.7)
    #test_matrix, rest = split_set(test_matrix, 0.5)
    #test_true, rest = split_set(test_true, 0.5)

    train_matrix = torch.tensor(train_matrix)
    test_matrix = torch.tensor(test_matrix)
    train_true = torch.tensor(train_true)
    test_true = torch.tensor(test_true)

    #train_loader = create_loader(train_matrix, train_true, batch_size)
    #test_loader = create_loader(test_matrix, test_true, batch_size)

    return train_matrix, train_true, test_matrix, test_true, station_names


def get_dataset(filename='../data/Hourly_NO2_referencedata_for_Oslo.csv', history=6):
    station_data = read_measurements(use_first=False, filename=filename)
    return predict_parallel_all(station_data, 'Bygdøy Alle', history)


if __name__ == '__main__':
    station_data = read_measurements(use_first=False, filename='Hourly_NO2_referencedata_for_Oslo.csv')
    predict_parallel_all(station_data, 'Bygdøy Alle', 64)
