from typing import List
from measure import Measure
import numpy as np
import random

"""
@staticmethod
def add_bias(X):
    sh = X.shape
    if len(sh) == 1:
        return np.concatenate([np.array([1]), X])
    else:
        m = sh[0]
        bias = np.ones((m, 1))
        return np.concatenate([bias, X], axis=1)
"""


def get_difference_in_hours(measure: Measure, result: Measure):
    measure_time = measure.datetime
    result_time = result.datetime
    diff = result_time - measure_time
    seconds = diff.total_seconds()
    return int(seconds / 60 / 60)


def build_simple_matrix_np(data: List[Measure], hours: int):
    matrix = np.empty((len(data) - hours, 6))
    true_value = []
    for index, measure in enumerate(data):
        result = get_result_value(data, measure, index, index + hours, hours)
        if result is None:
            continue
        matrix[index][0] = measure.value
        matrix[index][1] = measure.weekday
        matrix[index][2] = measure.hour
        matrix[index][3] = measure.month
        matrix[index][4] = measure.day
        matrix[index][5] = measure.year
        true_value.append(result)
    return matrix, np.array(true_value)


def build_simple_matrix(data: List[Measure], hours: int, include_low_cost=False):
    matrix = []
    true_value = []
    for index, measure in enumerate(data):
        result = get_result_values(data, measure, index, index + hours, hours)

        if result is None:
            continue
        if include_low_cost:
            matrix.append([measure.value,
                           measure.weekday,
                           measure.hour,
                           measure.month,
                           measure.day,
                           measure.year,
                           measure.station_type,
                           measure.is_low_cost])
        else:
            matrix.append([measure.value,
                           measure.weekday,
                           measure.hour,
                           measure.month,
                           measure.day,
                           measure.year,
                           measure.station_type])
        true_value.append(result)
    return matrix, true_value


def build_weather_matrix(data: List[Measure], hours: int, weather_data: List[str]):
    matrix = []
    true_value = []
    for index, measure in enumerate(data):
        result = get_result_values(data, measure, index, index + hours, hours)
        rain = float(weather_data[index])
        if result is None:
            continue
        matrix.append([measure.value,
                       measure.weekday,
                       measure.hour,
                       measure.month,
                       measure.day,
                       measure.year,
                       rain])
        true_value.append(result)
    return matrix, true_value


def build_simple_only_prev_matrix(data: List[Measure], hours: int, previous=5, is_stations=False):
    matrix = []
    true_value = []
    for index, measure in enumerate(data):
        if index < previous:
            continue
        result = get_result_values(data, measure, index, index + hours, hours)
        if result is None:
            continue
        count = 0
        new_element = []
        while previous > count:
            new_element.append(data[index - count].value)
            count += 1
        matrix.append(new_element)

        true_value.append(result)
    return matrix, true_value


def build_simple_prev_matrix(data: List[Measure], hours: int, previous=5):
    matrix = []
    true_value = []
    for index, measure in enumerate(data):
        if index < previous:
            continue
        result = get_result_value(data, measure, index, index + hours, hours)
        if result is None:
            continue
        count = 0
        new_element = [measure.value,
                       measure.weekday,
                       measure.hour,
                       measure.month,
                       measure.day,
                       measure.year]
        while previous > count:
            new_element.append(data[index - count].value)
            count += 1
        matrix.append(new_element)

        true_value.append(result)
    return matrix, true_value


def build_location_matrix(data: List[Measure], hours: int, include_low_cost=False):
    matrix = []
    true_value = []
    for index, measure in enumerate(data):
        result = get_result_value(data, measure, index, index + hours, hours)
        if result is None:
            continue
        if include_low_cost:
            matrix.append([measure.value,
                           measure.weekday,
                           measure.hour,
                           measure.month,
                           measure.day,
                           measure.year,
                           float(measure.latitude),
                           float(measure.longitude),
                           measure.is_low_cost])
        else:
            matrix.append([measure.value,
                           measure.weekday,
                           measure.hour,
                           measure.month,
                           measure.day,
                           measure.year])
        true_value.append(result)
    return matrix, true_value


def get_result_values(data: List[Measure], measure: Measure, original_index: int, index: int, hours: int):
    hour = 1
    values = []
    while hour <= hours:
        value = get_result_value(data, measure, original_index, index, hour)
        if value is None:
            # print('.')
            return None
        values.append(value)
        hour += 1
    return values


def get_result_value(data: List[Measure], measure: Measure, original_index: int, index: int, hours: int):
    result = get_measure(data, index)
    if result is None:
        return None
    difference = get_difference_in_hours(measure, result)
    if difference == hours:
        return result.value
    if difference > hours:
        reduce = difference - hours
        if original_index < index - reduce:
            result = get_result_value(data, measure, original_index, index - reduce, hours)
            return result
    return None


def get_measure(data: List[Measure], index: int) -> Measure:
    try:
        return data[index]
    except:
        return None


def shuffle_data(matrix, true_values):
    random.seed(2021)
    random.shuffle(matrix)
    random.seed(2021)
    random.shuffle(true_values)

    return matrix, true_values


def create_set(data, train_proportion):
    number_of_elements = len(data)
    train_size = int(number_of_elements * train_proportion)
    train = data[:train_size]
    test = data[train_size:]
    return train, test


"""
def create_set(data, chunk_size):
    number_of_elements = len(data)
    chucks = int(number_of_elements / chunk_size)
    train_size = int(chucks * (chunk_size - 2))
    train = data[:train_size]
    val = data[train_size:chucks + train_size]
    test = data[chucks + train_size:]
    return train, val, test
"""

