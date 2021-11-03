"""Old code repropreated for this"""
import numpy as np


def add_bias(X):
    sh = X.shape
    if len(sh) == 1:
        return np.concatenate([np.array([-1]), X])
    else:
        m = sh[0]
        bias = np.full((m, 1), -1)
        return np.concatenate([bias, X], axis=1)


class NumpyClassifier():
    def accuracy(self, x_test, y_test, **kwargs):
        predictions = self.predict(x_test, **kwargs)
        if len(predictions.shape) > 1:
            predictions = predictions[:, 0]
        return abs(predictions - y_test)


class NumpyLinReg(NumpyClassifier):
    def predict(self, x):
        z = add_bias(x)
        score = z @ self.weights
        return score

    def fit(self, x_train, t_train, eta=0.1, diff=1):
        (k, m) = x_train.shape
        x_train = add_bias(x_train)

        self.weights = weights = np.zeros(m + 1)

        # New code from here
        update = None
        self.count = 0
        while True:
            if update is not None:
                if all(abs(single_weight_update) < diff for single_weight_update in update):
                    break

            update = eta / k * x_train.T @ (x_train @ weights - t_train)
            weights -= update
            self.count += 1