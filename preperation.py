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
        mdf = mdf.drop('Close', axis=1)
        return mdf
