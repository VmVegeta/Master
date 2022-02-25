from typing import List, Dict

from measure import Measure
from readers.ReadMeasurments import read_measurements


def count_measurements(data: Dict[str, List[Measure]]):
    for station_name in data:
        print(station_name, len(data[station_name]))


if __name__ == '__main__':
    station_data = read_measurements(use_first=False, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')
    count_measurements(station_data)
