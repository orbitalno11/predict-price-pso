import pandas as pd
import numpy as np

class Preperation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_per_change(self) -> pd.DataFrame:
        self.df = self.df.drop('Date', axis=1)
        self.df['change'] = (
            (self.df.Close - self.df.Close.shift(1))/self.df.Close.shift(1))*100
        mdf = self.df.drop(0, axis=0)
        mdf = mdf.reset_index(drop=True)
        mdf = mdf.drop('Adj Close', axis=1)
        return mdf

    def getclose(self) -> float:
        last_close = self.df[-1:]
        return last_close['Close']
    
    def getcloseall(self, percent: np.ndarray, last_close: float):
        percent = percent[0]
        close = list()
        for i, value in enumerate(percent):
          print(value, i)
          if i == 0:
            close.append(((value/100) * last_close) + last_close)
          else :
            close.append(((value/100) * close[i-1]) + close[i-1])
        return close
