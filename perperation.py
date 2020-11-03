import pandas as pd


class Preperation:
    def __init__(self, df):
        self.df = df

    def calculate_per_change(self):
        df['percent_change'] = (
            (df.Close - df.Close.shift(1))/df.Close.shift(1))*100
        mdf = df.drop(0, axis=0)
        mdf = mdf.reset_index(drop=True)
        return mdf
