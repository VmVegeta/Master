from readers.ReadMeasurments import read_measurements
from Models.RNN import handle_station_dnn, handle_all_station_dnn, predict_all_stations_individually, predict_single_parallel, predict_parallel_all
#from Models.All_NN import handle_all_station_dnn, predict_all_stations_individually, predict_parallel, predict_parallel_closes, predict_parallel_all


def main():
    measurements = read_measurements(use_first=False)
    predict_all_stations_individually(measurements)
    #predict_parallel_all(measurements, 'Bygdøy Alle', 12)
    exit(0)
    handle_all_station_dnn(measurements)
    models = []
    for station_name in measurements:
        print(station_name)
        measurement = measurements[station_name]
        models.append(handle_all_station_dnn(measurement))
        exit(0)


if __name__ == '__main__':
    main()

