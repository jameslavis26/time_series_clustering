import numpy as np

class TimeSeriesData:
    def __init__(self, X, y=None, train_val_test_split=None, dataset_name=None, parameters=None, **kwargs):
        self.__dict__.update(kwargs)
        self.X = X
        self.y = y

        self.N = len(X)
        self.indices = np.arange(self.N)
        self.tvt_split = train_val_test_split
        self.dataset_name = dataset_name if dataset_name else ""
        self.parameters = parameters if parameters else {}

    def train_data(self):
        train_idx = self.indices[:int(self.tvt_split[0]*self.N)]

        if type(self.y) == type(None):
            return self.X[train_idx], None

        return self.X[train_idx], self.y[train_idx]

    def val_data(self, lag=0):
        val_idx = self.indices[int(self.tvt_split[1]*self.N) - lag:int(self.tvt_split[2]*self.N)]

        if type(self.y) == type(None):
            return self.X[val_idx], None

        return self.X[val_idx], self.y[val_idx]

    def test_data(self, lag=0):
        test_idx = self.indices[int(self.tvt_split[2]*self.N)-lag:]

        if type(self.y) == type(None):
            return self.X[test_idx], None

        return self.X[test_idx], self.y[test_idx]