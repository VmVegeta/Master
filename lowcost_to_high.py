import numpy as np
import tensorflow as tf
import os
from typing import List
from readers.ReadMeasurments import read_measurements
from readers.ReadLowcost import read_low_cost
from measure import Measure
from BuildMatrix import create_set
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tools.plot import plot_loss
from tools.custom_evaluation import coeff_determination


def get_input_result(input_measure: Measure, result_measure: Measure):
    matrix_element = [input_measure.value, input_measure.year, input_measure.month,
                      input_measure.hour, input_measure.weekday]
    result_element = result_measure.value
    return matrix_element, result_element


def construct_matrix(lowest_list: List[Measure], highest_list: List[Measure], lowest_is_low_cost: bool):
    matrix = []
    results = []
    lowest_length = len(lowest_list)
    highest_length = len(highest_list)
    lowest_count = 0
    highest_count = 0
    while lowest_length > lowest_count and highest_length > highest_count:
        lowest = lowest_list[lowest_count]
        highest = highest_list[highest_count]
        if lowest.day == highest.day and lowest.hour == highest.hour and lowest.month == highest.month and lowest.year == highest.year:
            lowest_count += 1
            highest_count += 1
            if lowest_is_low_cost:
                matrix_element, result_element = get_input_result(lowest, highest)
            else:
                matrix_element, result_element = get_input_result(highest, lowest)
            matrix.append(matrix_element)
            results.append(result_element)
        elif lowest.datetime < highest.datetime:
            lowest_count += 1
        else:
            highest_count += 1
    return matrix, results


def build_matrix(high_measurement: List[Measure], low_cost_measurement: List[Measure]):
    if high_measurement[0].datetime < low_cost_measurement[0].datetime:
        return construct_matrix(high_measurement, low_cost_measurement, False)
    else:
        return construct_matrix(low_cost_measurement, high_measurement, True)


def build_model(matrix, result, set_size=0.7, use_checkpoint=False):
    x_train, x_test = create_set(matrix, set_size)
    y_train, y_test = create_set(result, set_size)
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
        layers.Dropout(0.2),
        layers.Dense(1)
        # ,layers.Lambda(lambda x: tf.where(x < 0, tf.zeros_like(x), x), trainable=False)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[
                      coeff_determination,
                      keras.metrics.MeanAbsoluteError()
                  ])
    print(x_train.shape)
    # model.build(x_train.shape)
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=160)]

    if use_checkpoint:
        checkpoint_path = "checkpoints/nn_test2.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         model="val_loss",
                                                         mode="min",
                                                         save_best_only=True)
        checkpoint_path.format(epoch=0)
        callbacks.append(cp_callback)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("start_load")
        model.load_weights(latest)
        print("finish_load")


    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=0,
        epochs=5000,
        callbacks=callbacks
    )
    #
    print("original")
    print(model.evaluate(x_train, y_train, verbose=0))
    print("test")
    print(model.evaluate(x_test, y_test, verbose=0))
    #plot_loss(history)


def low_cost_to_high():
    high_measurement = read_measurements()
    low_cost_measurement = read_low_cost()

    hjortnes_station = high_measurement["Hjortnes"]
    hjortnes_low_cost = low_cost_measurement["1"]

    matrix, results = build_matrix(hjortnes_station, hjortnes_low_cost)
    build_model(matrix, results, True)

