from .time_series_base import TimeSeriesModel

class MovingAverageEstimator(TimeSeriesModel):
    def __init__(self, lag):
        super().__init__(lag=lag)

    def fit(self, x, y=None):
        pass

    def predict(self, x):
        x_test, _ = self.reshape_data(x)
        return x_test.mean(axis=1)