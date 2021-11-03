import csv
from measure import Measure


def read_measurements(use_first=False):
    measurements = {}
    current_measurement = None
    first = False
    with open("data/Hourly_NO2_referencedata_for_Oslo.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["QCFlagType(0=OK)"] != '0':
                continue

            station_name = row["Station name"]
            new_measure = Measure(
                station_name,
                row["value"],
                row["latitude"],
                row["longitude"],
                date=row["timeInUTC"],
                station_type=row["station type"])
            if current_measurement != station_name:
                if use_first:
                    if first:
                        break
                    first = True
                current_measurement = station_name
                measurements[station_name] = []
            measurements[station_name].append(new_measure)

    return measurements
