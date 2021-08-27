import pandas as pd

class DataframeAccumulator:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    def add(self, other: "DataframeAccumulator") -> "DataframeAccumulator":
        return DataframeAccumulator(pd.concat([self._df, other._df], ignore_index = True))
    
    def __add__(self, other):
        return add(self, other)

    def get(self) -> pd.DataFrame:
        return self._df

    def identity(self):
        return pd.DataFrame()