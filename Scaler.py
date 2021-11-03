class Scaler:
    def __init__(self, train_data):
        self._min = train_data.min(axis=0)
        self._max = train_data.max(axis=0)

    def scale(self, data):
        return (data - self._min) / (self._max - self._min)