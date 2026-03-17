import numpy as np

def to_lagged_time_series(x, y=None, lag=1):
    """
    Reshape the data into a rolling time series window
    """
    if len(x.shape) == 1:
        xt = np.lib.stride_tricks.sliding_window_view(x, window_shape=[lag])
    else:
        _, d = x.shape
        xt = np.lib.stride_tricks.sliding_window_view(x, window_shape=[lag, d])[:, 0, :, :]
    
    if type(y) != type(None):
        yt = y[lag-1:]
    else:
        yt = None

    return xt, yt

class TimeSeriesData:
    def __init__(self, X, y=None, lag=1, train_val_test_split=None, dataset_name=None, parameters=None, **kwargs):
        self.__dict__.update(kwargs)
        self.X = X
        self.y = y
        self.lag = lag

        self.N = len(X)
        self.indices = np.arange(self.N)
        self.tvt_split = train_val_test_split
        self.dataset_name = dataset_name if dataset_name else ""
        self.parameters = parameters if parameters else {}

    def train_data(self, lag=None, squeeze=False):
        if not lag:
            lag = self.lag

        min_idx = 0
        max_idx = int(self.tvt_split[0]*self.N)
        self.train_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.train_idx], None

        X, y = to_lagged_time_series(self.X[self.train_idx], self.y[self.train_idx], lag)
        if squeeze:
            return X.squeeze(), y.squeeze()
        return X, y

    def val_data(self, lag=None, squeeze=False):
        if not lag:
            lag = self.lag

        min_idx = int(self.tvt_split[0]*self.N) - lag
        max_idx = int((self.tvt_split[0] + self.tvt_split[1])*self.N)
        self.val_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.val_idx], None

        X, y = to_lagged_time_series(self.X[self.val_idx], self.y[self.val_idx], lag)
        if squeeze:
            return X.squeeze(), y.squeeze()
        return X, y
    
    def test_data(self, lag=None, squeeze=False):
        if not lag:
            lag = self.lag

        min_idx = int((self.tvt_split[0] + self.tvt_split[1])*self.N) - lag
        max_idx = self.N 
        self.test_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.test_idx], None

        X, y = to_lagged_time_series(self.X[self.test_idx], self.y[self.test_idx], lag)
        if squeeze:
            return X.squeeze(), y.squeeze()
        return X, y
    
    def full_data(self, lag=None, squeeze=False):
        if not lag:
            lag = self.lag
            
        X, y = to_lagged_time_series(self.X, self.y, lag)
        if squeeze:
            return X.squeeze(), y.squeeze()
        return X, y
    
    def drop_data(self):
        self.X = None
        self.y = None