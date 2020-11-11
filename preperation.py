import pandas as pd
import numpy as np
import pandas as pd


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
        return last_close['Close'].values[0]
    
    def getcloseall(self, percent: np.ndarray, last_close) -> pd.DataFrame:  
        percent = percent[0]
        close_list = list()
        for i, value in enumerate(percent):
          if i == 0:
            temp = ((value/100) * last_close) + last_close
            close_list.append(temp)

          else :
            temp = ((value/100) * close_list[i-1]) + close_list[i-1]
            close_list.append(temp)
            
        table_close_percont = pd.DataFrame(data=percent,columns=['percent'])
        table_close_percont['Close'] =close_list 
        return table_close_percont