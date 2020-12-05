from pandas import DataFrame
import numpy as np

from numpy.random import randn
# import model


class Simulator:

    def __init__(self, simulate_day: int, simulate_data: DataFrame, initial_fund: int = 1000000, trading_constant=0.5, volume_constant=100, trading_fee=0.17):
        self.simulate_day = simulate_day
        self.simulate_data = simulate_data
        self.fund = initial_fund
        self.TRADING_CONSTANT = trading_constant
        self.VOLUME_CONSTANT = volume_constant
        self.TRADING_FEE = trading_fee / 100
        self.predict_history = list()
        self.fund_history = list()
        self.stock_history = list()
        self.close_history = list()
        self.action_history = list()
        self.stock = 0

    def __trading(self, day: int):
        print("วันที่", day, " เงิน ", self.fund)
        predict = randn(1)[0]
        close = self.simulate_data.loc[day:day, ['Close']].values[0][0]
        if predict > self.TRADING_CONSTANT:
            self.__buy(close)
        elif predict < -self.TRADING_CONSTANT:
            self.__sell(close)
        else:
            self.__hold()
        self.close_history.append(close)
        self.predict_history.append(predict)

    def __buy(self, close):
        available_fund = self.fund / 3
        limit = int(np.round(available_fund/close)/100)*100
        if limit > 1:
            value = (limit * close) + self.TRADING_FEE
            self.fund = self.fund - value
            self.stock = self.stock + limit
            self.fund_history.append(self.fund)
            self.stock_history.append(self.stock)
            self.action_history.append("buy")
            print("fund ", self.fund)
            print("buy")
        else:
            self.__hold()

    def __sell(self, close):
        available_stock = self.stock / 3
        if available_stock >= 100:
            limit = int(available_stock/100)*100
            value = (limit * close) * (1 - self.TRADING_FEE)
            self.fund = self.fund + value
            self.stock = self.stock - limit
            self.fund_history.append(self.fund)
            self.stock_history.append(self.stock)
            self.action_history.append("sell")
            print("fund ", self.fund)
            print("sell")
        else:
            self.__hold()

    def __hold(self):
        self.fund_history.append(self.fund)
        self.stock_history.append(self.stock)
        self.action_history.append("hold")
        print('hold')

    def start(self):
        for day in range(self.simulate_day):
            self.__trading(day)
        return 'start'

    def summary(self):
        return DataFrame(list(zip(self.predict_history, self.close_history, self.fund_history, self.stock_history, self.action_history)), columns=['predict', 'close', 'fund', 'stock', 'action'])
