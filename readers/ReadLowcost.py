import csv

from measure import Measure


def read_low_cost(measurements=None):
    if measurements is None:
        measurements = {}
    current_measurement = None
    with open("../data/Hourly_NO2_lowcost_innosense_sensors_Oslo.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = row["instrumentid"]
            new_measure = Measure(
                station_id,
                row["grafNO2"],
                row["Latitude"],
                row["Longitude"],
                row["y"],
                row["m"],
                row["d"],
                row["h"])

            if current_measurement != station_id:
                current_measurement = station_id
                measurements[station_id] = []

            measurements[station_id].append(new_measure)

    return measurements
