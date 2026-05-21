
import pandas as pd

class DataIngestion:
    def __init__(self, path: str):
        self.path = path

    def load_data(self):
        return pd.read_csv(self.path)
