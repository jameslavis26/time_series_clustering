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
        min_idx = 0
        max_idx = int(self.tvt_split[0]*self.N)
        self.train_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.train_idx], None

        return self.X[self.train_idx], self.y[self.train_idx]

    def val_data(self, lag=0):
        min_idx = int(self.tvt_split[0]*self.N) - lag
        max_idx = int((self.tvt_split[0] + self.tvt_split[1])*self.N)
        self.val_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.val_idx], None

        return self.X[self.val_idx], self.y[self.val_idx]

    def test_data(self, lag=0):
        min_idx = int((self.tvt_split[0] + self.tvt_split[1])*self.N) - lag
        max_idx = self.N 
        self.test_idx = self.indices[min_idx:max_idx]

        if type(self.y) == type(None):
            return self.X[self.test_idx], None

        return self.X[self.test_idx], self.y[self.test_idx]
    
    def drop_data(self):
        self.X = None
        self.y = None