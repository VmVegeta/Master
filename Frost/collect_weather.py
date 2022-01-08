import subprocess
from readers.ReadMeasurments import read_measurements


def populate_weather_file(station_id, start_date, end_date):
    command = 'Rscript'
    rscript = 'C:/Users/vemun/PycharmProjects/Master/Frost/collect_weather.R'

    args = [station_id, '2018-01-01T00:00:00', '2019-01-01T00:00:00']
    cmd = [command, rscript] + args
    outputs = subprocess.check_output(cmd, universal_newlines=True)

    #outputs = station.split(',')
    #for output in outputs[0:3]:
    f = open("../data/Hjortnes_windspeed.txt", "a")
    f.write(outputs)
    f.close()


if __name__ == '__main__':
    measurements = read_measurements(filename="../data/Hourly_NO2_referencedata_for_Oslo.csv")
    start_date = measurements['Hjortnes'][0].datetime.__str__().replace(' ', 'T')
    end_date = measurements['Hjortnes'][len(measurements['Hjortnes']) - 1].datetime.__str__().replace(' ', 'T')
    populate_weather_file('SN18690', start_date, end_date)