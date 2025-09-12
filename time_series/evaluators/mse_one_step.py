import numpy as np
class MeanSquaredError:
    def __init__(self, X_test, y_test = None):
        if type(y_test) == type(None):
            self.X_test = X_test[:-1]
            self.y_test = X_test[1:]
        else:
            self.X_test = X_test
            self.y_test = y_test

    def __call__(self, trained_model):
        y_pred = trained_model.predict(self.X_test)
        lag_offset = len(self.y_test) - len(y_pred)

        score = np.mean((y_pred - self.y_test[lag_offset:])**2)
        return score