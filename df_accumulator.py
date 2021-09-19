import pandas as pd
from coffea.processor import AccumulatorABC

class DataframeAccumulator(AccumulatorABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    def add(self, other: "DataframeAccumulator") -> "DataframeAccumulator":
        return DataframeAccumulator(pd.concat([self._df, other._df], ignore_index = True))
    
    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)
        
    def get(self) -> pd.DataFrame:
        return self._df

    def identity(self):
        return DataframeAccumulator(pd.DataFrame())
    
    def concat(self, df: pd.DataFrame):
        return DataframeAccumulator(pd.concat([self._df, df], axis=1))
