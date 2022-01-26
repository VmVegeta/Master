import itertools
from datetime import timedelta
from measure import Measure
from typing import Dict, List
from readers.ReadMeasurments import read_measurements
import geopy.distance


def is_next_hour(prev: Measure, current: Measure):
    one_hour = prev.datetime + timedelta(hours=1)
    return one_hour == current.datetime


def test(data: Dict[str, List[Measure]]):
    for station_name in data:
        station_list = data[station_name]
        if len(station_list) < 40000:
            continue
        print(station_name, len(station_list), station_list[0].datetime, station_list[len(station_list) - 1].datetime)
        prev = station_list[0]
        print(prev.datetime)
        for measurement in station_list[1:]:
            if not is_next_hour(prev, measurement):
                print(prev.datetime, ',', measurement.datetime)
            prev = measurement


def get_distance(data: Dict[str, List[Measure]], station_name):
    starting_point = data[station_name][0]
    starting_corrdinate = (starting_point.latitude, starting_point.longitude)

    for station in data:
        measurement = data[station][0]
        measurement_corrdinate = (measurement.latitude, measurement.longitude)

        distance = geopy.distance.distance(starting_corrdinate, measurement_corrdinate).km
        print(station, distance)


def get_station_means(measurements: List[Measure]):
    sum = 0
    for measurement in measurements:
        sum += measurement.value
    return sum / len(measurements)


def build_closes_parallel_matrix(data: Dict[str, List[Measure]], to_predict, add_date=True):
    predict_measurements = data[to_predict]
    input_data = []
    means = []
    stations = ['Hjortnes', 'Kirkeveien', 'Smestad']

    for key in data:
        if key not in stations:
            continue
        measurements = data[key]
        means.append(get_station_means(measurements))
        input_data.append(iter(measurements))

    matrix = []
    true_value = []
    for predict_measurement in predict_measurements:
        current_datetime = predict_measurement.datetime
        single_input = []
        keep = False
        for index, input_measurements in enumerate(input_data):
            while True:
                measurement = next(input_measurements, None)
                if measurement is None:
                    single_input.append(means[index])
                    break
                if current_datetime == measurement.datetime:
                    single_input.append(measurement.value)
                    keep = True
                    break
                if current_datetime < measurement.datetime:
                    itertools.chain([measurement], input_measurements)
                    single_input.append(means[index])
                    break
                next(input_measurements)
        if keep:
            #single_input.append(predict_measurement.hour)
            #single_input.append(predict_measurement.weekday)
            #single_input.append(predict_measurement.month)
            #single_input.append(predict_measurement.year)
            matrix.append(single_input)
            true_value.append(predict_measurement.value)
    print(len(true_value))
    return matrix, true_value


def build_parallel_matrix(data: Dict[str, List[Measure]], to_predict, add_date=True):
    predict_measurements = data[to_predict]
    input_data = []
    means = []

    for key in data:
        measurements = data[key]
        if key == to_predict or len(measurements) < 40000:
            continue
        #measurements = measurements.sort(key=lambda x: x.datetime)
        means.append(get_station_means(measurements))
        input_data.append(iter(measurements))

    matrix = []
    true_value = []
    for predict_measurement in predict_measurements:
        current_datetime = predict_measurement.datetime
        single_input = []
        keep = 0
        #keep = True
        for index, input_measurements in enumerate(input_data):
            while True:
                measurement = next(input_measurements, None)
                if measurement is None:
                    single_input.append(means[index])
                    keep += 1
                    break
                if current_datetime == measurement.datetime:
                    #keep += 1
                    single_input.append(measurement.value)
                    break
                if current_datetime < measurement.datetime:
                    itertools.chain([measurement], input_measurements)
                    single_input.append(means[index])
                    keep += 1
                    break
                next(input_measurements)
        if keep < 3:
            single_input.append(predict_measurement.hour)
            single_input.append(predict_measurement.weekday)
            single_input.append(predict_measurement.month)
            single_input.append(predict_measurement.year)
            matrix.append(single_input)
            true_value.append(predict_measurement.value)
    print(len(true_value))
    return matrix, true_value


def get_parallel_matrix(station_name: str):
    station_data = read_measurements(use_first=False, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')
    return build_parallel_matrix(station_data, station_name)


if __name__ == '__main__':
    station_data = read_measurements(use_first=False, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')
    get_distance(station_data, 'Bygdøy Alle')
    #test(station_data)
