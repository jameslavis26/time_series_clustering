import numpy as np

def to_time_series(x, y=None, lag=1):
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