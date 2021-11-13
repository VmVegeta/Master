import subprocess
from readers.ReadMeasurments import read_measurements


def test(station_name, lat, long):
    command = 'Rscript'
    rscript = 'C:/Users/vemun/PycharmProjects/Master/Frost/test.R'

    args = [lat, long]
    cmd = [command, rscript] + args
    station = subprocess.check_output(cmd, universal_newlines=True)

    outputs = station.split(',')
    distance = float(outputs[3])
    print(station_name, outputs[1], outputs[2], outputs[0], distance < 0.25)


if __name__ == '__main__':
    measurements = read_measurements(filename="../data/Hourly_NO2_referencedata_for_Oslo.csv")
    for station_name in measurements:
        test(station_name, str(measurements[station_name][0].latitude), str(measurements[station_name][0].longitude))

