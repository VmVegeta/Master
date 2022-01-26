from matrixes.build_parallel_matrix import get_parallel_matrix
from matrixes.BuildMatrix import build_simple_matrix, shuffle_data, create_set, build_simple_prev_matrix
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error


def linear_run(station_name, set_size):
    matrix, true_value = get_parallel_matrix(station_name)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val = create_set(matrix, set_size)
    y_train, y_val = create_set(true_value, set_size)

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    print(reg.score(x_val, y_val))


def polynomial_features(x_train, y_train, x_val, y_val, degree):
    poly = PolynomialFeatures(degree=degree)
    fitted = poly.fit_transform(x_train)
    reg = linear_model.LinearRegression()
    reg.fit(fitted, y_train)
    print(reg.score(poly.transform(x_val), y_val))


def polly_run(station_name: str, set_size: float, degree=2):
    matrix, true_value = get_parallel_matrix(station_name)
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_val = create_set(matrix, set_size)
    y_train, y_val = create_set(true_value, set_size)
    degrees = [2, 4, 6]
    #for degree in degrees:
    polynomial_features(x_train, y_train, x_val, y_val, degree)


if __name__ == '__main__':
    linear_run('Manglerud', 0.7)