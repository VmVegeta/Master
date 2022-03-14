

def main(new_filename: str, station_one: str, station_two: str, filename="../data/Hourly_NO2_referencedata_for_Oslo.csv"):
    new_file = open(new_filename, "w", encoding="utf-8")

    with open(filename, mode="r", encoding="utf-8") as f:
        new_file.write(f.readline())
        for row in f.readlines():
            if station_one in row:
                new_file.write(row)
            if station_two in row:
                new_file.write(row)


if __name__ == "__main__":
    main('../data/Alnabru_Bygdøy.csv', 'Alnabru', 'Bygdøy Alle')
