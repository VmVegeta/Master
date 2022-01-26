from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from matrixes.BuildMatrix import build_simple_matrix, shuffle_data, create_set, build_simple_prev_matrix
from measure import Measure
from collections import defaultdict
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error


def plot_time(measurements, station_name):
    date_times = [measurement.datetime for measurement in measurements]
    values = [measurement.value for measurement in measurements]
    plt.title = station_name
    print(station_name)
    plt.plot_date(date_times, values)
    plt.show()


def plot_all(measurements):
    for station_name in measurements:
        plot_time(measurements[station_name], station_name)


def separate_on_station(measurements):
    station_measurements = defaultdict(list)
    for measurement in measurements:
        station_measurements[measurement.station_id].append(measurement)

    return station_measurements


def plot_result(model, matrix, true_values, poly=None):
    if poly is not None:
        matrix = poly.transform(matrix[1000:2000])
    else:
        matrix = matrix[1000:2000]
    predictions = model.predict(matrix)
    plt.plot(true_values[1000:2000], label="True")
    plt.plot(predictions, label="Pred")
    plt.legend()
    plt.show()


def polynomial_features(x_train, y_train, x_val, y_val, degree):
    print("PolynomialFeatures - " + str(degree))
    poly = PolynomialFeatures(degree=degree)
    fitted = poly.fit_transform(x_train)
    reg = linear_model.LinearRegression()
    reg.fit(fitted, y_train)
    print(reg.score(poly.transform(x_val), y_val))
    return reg, poly


def linear_run(station_measurements: List[Measure], set_size=8, hours=6):
    matrix, true_value = build_simple_prev_matrix(station_measurements, hours)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val, x_test = create_set(matrix, set_size)
    y_train, y_val, y_test = create_set(true_value, set_size)

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    print(reg.score(x_val, y_val))


def polly_run(station_measurements: List[Measure], set_size=8, hours=6):
    """
    for previous in [2, 4, 8]:
        print(previous)
    """
    matrix, true_value = build_simple_matrix(station_measurements, hours=hours)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val, x_test = create_set(matrix, set_size)
    y_train, y_val, y_test = create_set(true_value, set_size)
    degrees = [2, 4, 6]
    for degree in degrees:
        model, poly = polynomial_features(x_train, y_train, x_val, y_val, degree)
    plot_result(model, matrix, true_value, poly=poly)


def MLP_run(station_measurements: List[Measure], set_size=8, hours=6):
    matrix, true_value = build_simple_matrix(station_measurements, hours)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val, x_test = create_set(matrix, set_size)
    y_train, y_val, y_test = create_set(true_value, set_size)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)

    scores = []
    iteration = 1
    for x in range(iteration):
        regr = MLPRegressor(max_iter=1500).fit(scaled, y_train)
        # print(x_test[0][0])
        # print(regr.predict([x_test[0]]))
        # print(y_test[0])
        # print(regr.score(x_val, y_val))
        print(x)
        scores.append(regr.score(scaler.transform(x_val), y_val))
    print(scores)
    print(sum(scores)/iteration)
    mse = mean_squared_error(y_val, regr.predict(x_val))
    print(mse)
    #plot_result(regr, matrix, true_value, poly=scaler)


def tree_run(station_measurements: List[Measure], set_size=8, hours=6):
    matrix, true_value = build_simple_prev_matrix(station_measurements, hours)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val, x_test = create_set(matrix, set_size)
    y_train, y_val, y_test = create_set(true_value, set_size)

    regr = DecisionTreeRegressor()
    iteration = 10
    scores = []
    for x in range(iteration):
        regr.fit(x_train, y_train)
        scores.append(regr.score(x_val, y_val))
    print(scores)
    print(sum(scores) / iteration)
    plot_result(regr, matrix, true_value)


def forest_run(station_measurements: List[Measure], set_size=8, hours=6):
    matrix, true_value = build_simple_matrix(station_measurements, hours)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val, x_test = create_set(matrix, set_size)
    y_train, y_val, y_test = create_set(true_value, set_size)

    iteration = 10
    forest = []
    for x in range(iteration):
        regr = DecisionTreeRegressor()
        regr.fit(x_train, y_train)
        forest.append(regr)
    predictions = []
    for tree in forest:
        predictions.append(tree.predict(x_val))

    presum = np.sum(predictions, axis=0)
    actual_prediction = presum / iteration
    y_val = np.asarray(y_val)
    v = ((y_val - y_val.mean()) ** 2).sum()
    u = ((y_val - actual_prediction) ** 2).sum()
    score = 1 - (u / v)
    print(score)
    plot_result(regr, matrix, true_value)


def old_run(measurements: dict):
    # plot_all(measurements)
    for station_name in measurements:
        print(station_name)
        station_measurements = measurements[station_name]
        # polly_run(station_measurements, hours=1)
        # linear_run(station_measurements)
        MLP_run(station_measurements)
        # tree_run(station_measurements)
        # forest_run(station_measurements)

        """
        scaled = False
        if scaled:
            scaler = StandardScaler()
            fitted_data = scaler.fit_transform(x_train)

            print("LinearRegression - True")
            reg = linear_model.LinearRegression()
            reg.fit(fitted_data, y_train)
            print(reg.score(scaler.transform(x_val), y_val))
        """
