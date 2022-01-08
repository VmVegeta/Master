import numpy as np
import tensorflow as tf
import os
from BuildMatrix import build_simple_matrix, shuffle_data, create_set, build_weather_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tools.plot import plot_loss
from tools.custom_evaluation import coeff_determination


def base_dnn(matrix, true_value, hours=6, set_size=0.7, use_checkpoint=False, load_weights=False):
    x_train, x_test = create_set(matrix, set_size)
    y_train, y_test = create_set(true_value, set_size)
    x_val, x_test = create_set(x_test, 0.5)
    y_val, y_test = create_set(y_test, 0.5)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(x_train)

    model = keras.Sequential([
          normalizer,
          layers.Dense(64, activation='relu'),
          layers.Dropout(0.2),
          layers.Dense(64, activation='relu'),
          layers.Dense(64, activation='relu'),
          #layers.Dropout(0.2),
          layers.Dense(hours)
          #,layers.Lambda(lambda x: tf.where(x < 0, tf.zeros_like(x), x), trainable=False)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[
                      coeff_determination,
                      keras.metrics.MeanAbsoluteError()
                  ])
    print(x_train.shape)
    model.build(x_train.shape)
    model.summary()
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)]

    if use_checkpoint is True:
        checkpoint_path = "checkpoints/nn_all3x64.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         model="val_loss",
                                                         mode="min",
                                                         save_best_only=True)
        if load_weights:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            model.load_weights(latest)
        checkpoint_path.format(epoch=0)
        callbacks.append(cp_callback)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=0,
        epochs=5000,
        callbacks=callbacks
    )

    print("original")
    print(model.evaluate(x_train, y_train, verbose=0))
    print("test")
    print(model.evaluate(x_test, y_test, verbose=0))
    plot_loss(history)
    return model, x_train, y_train, x_test, y_test


def handle_all_station_dnn(station_data, hours=6, set_size=0.7):
    matrix, true_value = [], []
    f = open("data/Alnabru_percipitation.txt", "r")
    weather_data = f.read()
    f.close()
    weather_data = weather_data.split(',')
    for station_name in station_data:
        station_matrix, station_true_value = build_weather_matrix(station_data[station_name], hours, weather_data)
        for matrix_input in station_matrix:
            matrix.append(matrix_input)
        for true_value_input in station_true_value:
            true_value.append(true_value_input)
    return base_dnn(matrix, true_value, hours, set_size)


def predict_all_stations_individually(station_data, hours=6, set_size=0.7):
    matrix, true_value = [], []
    for station_name in station_data:
        station_matrix, station_true_value = build_simple_matrix(station_data[station_name], hours)
        for matrix_input in station_matrix:
            matrix.append(matrix_input)
        for true_value_input in station_true_value:
            true_value.append(true_value_input)
    output = base_dnn(matrix, true_value, hours, set_size)

    for station_name in station_data:
        print(station_name)
        station_matrix, station_true_value = build_simple_matrix(station_data[station_name], hours)
        print(output[0].evaluate(station_matrix, station_true_value, verbose=0))
