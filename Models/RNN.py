import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from matrixes.BuildMatrix import build_simple_matrix, build_simple_only_prev_matrix, shuffle_data, create_set
from matrixes.build_parallel_matrix import build_single_parallel_matrix
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def plot_loss(history):
      plt.plot(history.history['loss'], label='loss')
      plt.plot(history.history['val_loss'], label='val_loss')
      plt.xlabel('Epoch')
      plt.ylabel('Error [MPG]')
      plt.legend()
      plt.grid(True)


def base_dnn(matrix, true_value, hours=6, set_size=0.5, use_checkpoint=False):
    matrix, true_value = shuffle_data(matrix, true_value)

    x_train, x_test = create_set(matrix, set_size)
    y_train, y_test = create_set(true_value, set_size)
    x_val, x_test = create_set(x_test, 0.5)
    y_val, y_test = create_set(y_test, 0.5)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = keras.Sequential([
        layers.Embedding(input_dim=x_train.shape[0], input_length=x_train.shape[1], output_dim=30),
        layers.LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(hours)
        # ,layers.Lambda(lambda x: tf.where(x < 0, tf.zeros_like(x), x), trainable=False)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[
                      coeff_determination,
                      keras.metrics.MeanAbsoluteError()
                  ])

    model.build(x_train.shape)
    model.summary()
    cp_callback = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)]

    if use_checkpoint:
        checkpoint_path = "rnn_checkpoints/actual_all1.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         model="val_loss",
                                                         mode="min",
                                                         save_best_only=True))
        checkpoint_path.format(epoch=0)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=0,
        epochs=1000,
        callbacks=cp_callback
    )

    plot_loss(history)
    print("original")
    print(model.evaluate(x_train, y_train, verbose=0))
    print("test")
    print(model.evaluate(x_test, y_test, verbose=0))

    #model.save("saved_model/All_RNN")

    return model, x_train, y_train, x_test, y_test


def handle_station_dnn(station_data, hours=6, set_size=0.7):
    matrix, true_value = build_simple_matrix(station_data, hours)
    return base_dnn(matrix, true_value, hours, set_size)


def handle_all_station_dnn(station_data, hours=6, set_size=0.6):
    matrix, true_value = [], []
    for station_name in station_data:
        station_matrix, station_true_value = build_simple_only_prev_matrix(station_data[station_name], hours, previous=30)
        for matrix_input in station_matrix:
            matrix.append(matrix_input)
        for true_value_input in station_true_value:
            true_value.append(true_value_input)
    return base_dnn(matrix, true_value, hours, set_size)


def predict_all_stations_individually(station_data, hours=6, set_size=0.6):
    matrix, true_value = [], []
    for station_name in station_data:
        station_matrix, station_true_value = build_simple_only_prev_matrix(station_data[station_name], hours,
                                                                           previous=30)
        for matrix_input in station_matrix:
            matrix.append(matrix_input)
        for true_value_input in station_true_value:
            true_value.append(true_value_input)
    output = base_dnn(matrix, true_value, hours, set_size)

    for station_name in station_data:
        print(station_name)
        station_matrix, station_true_value = build_simple_only_prev_matrix(station_data[station_name], hours,
                                                                           previous=30)
        print(output[0].evaluate(station_matrix, station_true_value, verbose=0))


def predict_single_parallel(station_data, station_name, from_station, hours=6):
    matrix, true_values = build_single_parallel_matrix(station_data[station_name], station_data[from_station], hours)
    base_dnn(matrix, true_values, 1, use_checkpoint=False)


def predict_parallel_all(station_data: str, station_name: str, hours=6):
    result_data = station_data[station_name]
    for measurement_station in station_data:
        if measurement_station == station_name:
            continue
        input_data = station_data[measurement_station]
        print(measurement_station)
        matrix, true_values = build_single_parallel_matrix(result_data, input_data, hours)
        if len(matrix) != 0:
            base_dnn(matrix, true_values, 1, use_checkpoint=False)
