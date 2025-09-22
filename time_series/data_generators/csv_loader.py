import pandas as pd
class CSVData:
    def __init__(self, filepath):
        self.filepath = filepath

    def __call__(self):
        data = pd.read_csv(self.filepath, index_col=0).values
        return 0, data