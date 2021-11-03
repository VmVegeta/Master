from readers.ReadMeasurments import read_measurements
from Models.RNN import handle_station_dnn, handle_all_station_dnn
#from Models.All_NN import handle_all_station_dnn


def main():
    measurements = read_measurements(use_first=True)
    # keras.backend.clear_session()
    handle_all_station_dnn(measurements)
    exit(0)

    models = []
    for station_name in measurements:
        print(station_name)
        measurement = measurements[station_name]
        models.append(handle_all_station_dnn(measurement))
        exit(0)


if __name__ == '__main__':
    main()

