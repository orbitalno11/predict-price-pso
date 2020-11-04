from pandas import DataFrame
import numpy as np


class Indicator:

    def __init__(self, dataset: DataFrame, rsi_n_day: int = 14):
        self.__dataset = dataset.copy()
        self.__rsi_n_day = rsi_n_day

    def __rsi_avg_gain(self, dataframe: DataFrame) -> float:
        previous_rs = dataframe.iloc[[-1]]['rs'].values[0]

        if np.isnan(previous_rs):
            gain = dataframe.loc[dataframe['change'] >= 0]
            avg_gain = np.absolute(gain['change'].sum() / self.__rsi_n_day)
            return avg_gain
        else:
            previous_avg_gain = dataframe.iloc[[-1]]['rsi_avg_gain'].values[0]
            _gain = dataframe.iloc[[-1]]['change'].values[0]
            gain = _gain if _gain >= 0 else 0
            avg_gain = ((previous_avg_gain * (self.__rsi_n_day - 1)) + gain) / self.__rsi_n_day
            return avg_gain

    def __rsi_avg_loss(self, dataframe: DataFrame) -> float:
        previous_rs = dataframe.iloc[[-1]]['rs'].values[0]

        if np.isnan(previous_rs):
            loss = dataframe.loc[dataframe['change'] < 0]
            avg_loss = np.absolute(loss['change'].sum() / self.__rsi_n_day)
            return avg_loss
        else:
             _avg_loss = dataframe.iloc[[-1]]['rsi_avg_loss'].values[0]
            _loss = dataframe.iloc[[-1]]['change'].values[0]
            loss = np.absolute(_loss) if _loss < 0 else 0
            avg_loss = ((previous_avg_loss * (self.__rsi_n_day - 1)) + loss) / self.__rsi_n_day
            return avg_loss

    def __RS(self, row_index: int) -> float:
        dataframe = self.__dataset[row_index - self.__rsi_n_day: row_index]
        avg_gain = self.__rsi_avg_gain(dataframe)
        avg_loss = self.__rsi_avg_loss(dataframe)
        rs = avg_gain / avg_loss

        self.__dataset.loc[row_index, ['rsi_avg_gain']] = avg_gain
        self.__dataset.loc[row_index, ['rsi_avg_loss']] = avg_loss
        self.__dataset.loc[row_index, ['rs']] = rs

        return rs

    def RSI(self) -> DataFrame:
        self.__dataset['change'] = self.__dataset['Close'] - self.__dataset['Close'].shift(1)
        self.__dataset['rsi_avg_gain'] = np.nan
        self.__dataset['rsi_avg_loss'] = np.nan
        self.__dataset['rs'] = np.nan

        for index in self.__dataset.index:
            if index > self.__rsi_n_day:
                self.__RS(index)

        self.__dataset['rsi'] = 100 - (100 / (1 + self.__dataset['rs']))
        return self.__dataset

    def __SMA(self,EMA_n_day:int) -> float:
        SMA=(self.__dataset.loc[:EMA_n_day-1,['Close']].sum().values[0])/EMA_n_day
        return SMA

    def EMA(self) -> DataFrame:
        EMA_n_day = 5
        self.__dataset['EMA'] = np.nan
        self.__dataset.loc[EMA_n_day-1,['EMA']] = self.__SMA(EMA_n_day)

        for index in self.__dataset.index: 
            if index > (EMA_n_day-1):
                previous_EMA = self.__dataset.iloc[[index-1]]['EMA'].values[0]
                EMA = previous_EMA + (2/EMA_n_day+1)*(self.__dataset.iloc[[index]]['Close'].values[0]-previous_EMA)
                self.__dataset.loc[index,['EMA']] = EMA

        return self.__dataset

    def MACD(self) -> str:
        return "MACD"
