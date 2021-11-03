import numpy as np
import tensorflow as tf
import os
from BuildMatrix import build_simple_matrix, shuffle_data, create_set
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tools.plot import plot_loss
from tools.custom_evaluation import coeff_determination


def handle_all_station_dnn(station_data, hours=6, set_size=0.7):
    matrix, true_value = build_simple_matrix(station_data, hours)
    matrix, true_value = shuffle_data(matrix, true_value)

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
          layers.Dense(256, activation='relu'),
          layers.Dropout(0.2),
          layers.Dense(64, activation='relu'),
          layers.Dropout(0.2),
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
    # model.build(x_train.shape)
    model.summary()

    checkpoint_path = "checkpoints/nn_test2.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=0,
                                                     model="val_loss",
                                                     mode="min",
                                                     save_best_only=True)
    checkpoint_path.format(epoch=0)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=0,
        epochs=5000,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=160), cp_callback]
    )

    #
    print("original")
    print(model.evaluate(x_train, y_train, verbose=0))
    print("test")
    print(model.evaluate(x_test, y_test, verbose=0))
    plot_loss(history)
    return model, x_train, y_train, x_test, y_test


