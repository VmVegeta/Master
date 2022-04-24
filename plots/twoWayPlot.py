import seaborn as sns
import matplotlib.pyplot as plt
from readers.ReadMeasurments import read_measurements
from matrixes.BuildMatrix import build_weather_matrix
import pandas as pd
import numpy as np
import datetime
from tools import count_img_pixels
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def main():
    station_data = read_measurements(use_first=True, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')['Alnabru']
    f = open("../data/Alnabru_percipitation.txt", "r")
    weather_data = f.read()
    f.close()
    weather_data = weather_data.split(',')
    matrix, true_value = build_weather_matrix(station_data, 1, weather_data)
    matrix = np.row_stack(matrix)
    df = pd.DataFrame(matrix, columns=['value', 'weekday', 'hour', 'month', 'day', 'year', 'rain'])
    sns.pairplot(df,
        x_vars=['rain'],
        y_vars=["value"])
    create_line(df['rain'].values, df['value'].values)
    plt.show()
    #display(df)


def windspeed_hjortness():
    station_data = read_measurements(use_first=False, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')[
        'Hjortnes']
    f = open("../data/Hjortnes_windspeed.txt", "r")
    weather_data = f.read()
    f.close()
    weather_data = weather_data.split(';')
    current_time = station_data[0].datetime
    last_value = 0
    values = []
    for weather in weather_data[:-1]:
        date_value = weather.split(',')
        weather_time = datetime.datetime.strptime(date_value[0], '%Y-%m-%dT%H:%M:%S.%fZ')
        while weather_time > current_time:
            values.append(last_value)
            current_time += datetime.timedelta(hours=1)
        last_value = date_value[1]

    length = len(values)
    matrix, true_value = build_weather_matrix(station_data[:length], 0, values)
    matrix = np.row_stack(matrix)
    df = pd.DataFrame(matrix, columns=['value', 'weekday', 'hour', 'month', 'day', 'year', 'wind'])
    #sns.pairplot(df, x_vars=['wind'], y_vars=['value'])
    plt.scatter(df['wind'], df['value'], edgecolor='white')
    plt.title("Wind Speed and Air Pollution Relation W. Regression Line")
    plt.xlabel("Wind Speed m/s")
    plt.ylabel("NO2 µg/m³")
    create_line(df['wind'].values, df['value'].values, degree=1)
    plt.show()


def two_way_plot_driving():
    root_path = "C:/Users/vemun/PycharmProjects/Master/osmaug/data/tile/"
    red_count = []
    for index in range(1, 15):
        red_count.append(count_img_pixels.main(root_path + str(index) + "_h_adt_zoom_17.jpg"))

    print(red_count)
    # station_data = read_measurements(use_first=False, filename='../data/Hourly_NO2_referencedata_for_Oslo.csv')
    means = [37.050250274871246, 27.48573612099576, 20.493297229274056, 38.15526019800008, 37.61102810441892,
             27.562392747892265, 37.71383188508532, 25.915224257967775, 26.486888457860466, 26.705612817874748,
             35.75284001442319, 37.75276580446405, 13.0871673509759, 29.95582040113164]
    """
    for station_name in station_data:
        station_val = station_data[station_name]
        value_list = [mes.value for mes in station_val]
        means.append(statistics.mean(value_list))
    print(means)
    """
    plt.scatter(red_count, means)
    plt.show()


def create_line(x, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    x = x.reshape(-1, 1)
    fitted = poly.fit_transform(x)

    lin = linear_model.LinearRegression()
    lin.fit(fitted, y)
    sorted_values = sorted(x)
    line = np.linspace(sorted_values[0], sorted_values[len(sorted_values) - 1], 1000)
    pred = lin.predict(poly.transform(line))
    plt.plot(line, pred, color="red")


if __name__ == '__main__':
    windspeed_hjortness()
