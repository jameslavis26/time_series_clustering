import numpy as np

class TimeSeriesModel:
    def __init__(self, lag):
        self.lag = lag

    def reshape_data(self, x, y=None):
        """
        Reshape the data into a rolling time series window
        """
        if len(x.shape) == 1:
            xt = np.lib.stride_tricks.sliding_window_view(x, window_shape=[self.lag])
        else:
            _, d = x.shape
            xt = np.lib.stride_tricks.sliding_window_view(x, window_shape=[self.lag, d])[:, 0, :, :]
        
        if type(y) != type(None):
            yt = y[self.lag-1:]
        else:
            yt = None

        return xt, yt