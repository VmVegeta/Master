import datetime


class Measure:
    def __init__(self, station_id, value, latitude, longitude, date=None, station_type=None, year=0, month=0, day=0, hour=0):
        self.station_id = station_id
        self.value = float(value)
        if self.value < 0:
            self.value = 0
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        if station_type is None:
            print('.')
            self.station_type = 0
            self.year = int(year)
            self.month = int(month)
            self.day = int(day)
            self.hour = int(hour)
            self.is_low_cost = 1
            self.datetime = datetime.datetime(self.year, self.month, self.day, self.hour)
        else:
            if station_type == 'Near Road station':
                self.station_type = 1
            else:
                self.station_type = 0
            date_time = self.get_date_time(date)
            self.datetime = date_time
            self.day = date_time.day
            self.year = date_time.year
            self.month = date_time.month
            self.hour = date_time.hour
            self.is_low_cost = 0
        self.weekday = self.datetime.weekday()
        """
        self.datetime = date_time.date()
        self.time = date_time.time()"""

    @staticmethod
    def get_date_time(date: str):
        if date is None:
            print("Date is None")
            return None
        return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def station_type(station_type: str) -> int:
        if station_type == "Near Road station":
            return 1
        else:
            print(station_type)
        return 0
