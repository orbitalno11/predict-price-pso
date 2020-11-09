from pandas import DataFrame
import numpy as np


class Indicator:

    def __init__(self, dataset: DataFrame, rsi_n_day: int = 14, ema_n_day: int = 5, sma_n_day: int = 5):
        self.__dataset = dataset.copy()
        self.__rsi_n_day = rsi_n_day
        self.__ema_n_day = ema_n_day
        self.__sma_n_day = sma_n_day

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
            previous_avg_loss = dataframe.iloc[[-1]]['rsi_avg_loss'].values[0]
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

    def SMA(self, sma_n_day: int = None) -> float:
        self.__sma_n_day = sma_n_day if sma_n_day != None else self.__sma_n_day
        SMA = (self.__dataset.loc[:self.__sma_n_day - 1]['Close'].sum()) / self.__sma_n_day
        return SMA

    def EMA(self, EMA_n_day: int = None) -> DataFrame:
        self.__ema_n_day = EMA_n_day if EMA_n_day != None else self.__ema_n_day
        column_name = 'ema_{}_day'.format(self.__ema_n_day)
        self.__dataset[column_name] = np.nan
        self.__dataset.loc[self.__ema_n_day - 1, [column_name]] = self.SMA(self.__ema_n_day)

        for index in self.__dataset.index:
            if index > (self.__ema_n_day - 1):
                previous_EMA = self.__dataset.iloc[[index - 1]][column_name].values[0]
                EMA = previous_EMA + (2 / self.__ema_n_day + 1) * (
                        self.__dataset.iloc[[index]]['Close'].values[0] - previous_EMA)
                self.__dataset.loc[index, [column_name]] = EMA
        return self.__dataset

    def MACD(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> DataFrame:
        self.EMA(fast_period)
        self.EMA(slow_period)
        self.__dataset['MACD'] = np.nan
        self.__dataset['signal'] = np.nan
        column_slow_period = 'ema_{}_day'.format(slow_period)
        column_fast_period = 'ema_{}_day'.format(fast_period)
        self.__dataset['MACD'] = self.__dataset[column_slow_period] - self.__dataset[column_fast_period]

        first_ema = self.__dataset.loc[slow_period - 1:(slow_period - 1) + (signal_period - 1)][
                        'MACD'].sum() / signal_period
        self.__dataset.loc[(slow_period - 1) + (signal_period - 1), 'signal'] = first_ema

        for index in self.__dataset.index:
            if index > (slow_period - 1) + (signal_period - 1):
                previous_EMA = self.__dataset.iloc[[index - 1]]['signal'].values[0]
                EMA = previous_EMA + (2 / signal_period + 1) * (
                        self.__dataset.iloc[[index]]['MACD'].values[0] - previous_EMA)
                self.__dataset.loc[index, ['signal']] = EMA

        self.__dataset['Histogram'] = self.__dataset['MACD'] - self.__dataset['signal']

        return self.__dataset
