from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from numpy.random import randn

from preparation import Preparation


# import model


class Simulator:

    def __init__(self, simulate_day: int, simulate_data: DataFrame, baseline_model, pso_model,
                 initial_fund: int = 1000000, trading_constant=0.5,
                 volume_constant=100, trading_fee=0.17):
        self.simulate_day = simulate_day
        self.simulate_data = simulate_data
        self.__prepare_data(simulate_data)
        self.TRADING_CONSTANT = trading_constant
        self.VOLUME_CONSTANT = volume_constant
        self.TRADING_FEE = trading_fee / 100
        # initial model
        self.baseline_model = baseline_model
        self.pso_model = pso_model
        # initial fund
        self.fund = initial_fund
        self.pso_fund = initial_fund
        # initial stock
        self.stock = 0
        self.pso_stock = 0
        # initial predict history
        self.baseline_predict_history = [0]
        self.pso_optimize_predict_history = [0]
        # initial fund history
        self.fund_history = [self.fund]
        self.pso_fund_history = [self.pso_fund]
        # initial stock history
        self.stock_history = [self.stock]
        self.pso_stock_history = [self.pso_stock]
        # initial action history
        self.action_history = ["start"]
        self.pso_action_history = ["start"]
        # initial close history
        self.close_history = [np.nan]

    def __prepare_data(self, dataframe: DataFrame):
        preparation = Preparation(dataframe)
        preparation_data = preparation.calculate_per_change()
        self.prepare_data = preparation_data

    def __trading(self, day: int):
        current_close = self.simulate_data.loc[day:day, ['Close']].values[0][0]
        data_for_predict = self.simulate_data.loc[day:day, ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(data_for_predict)
        baseline_predict = randn(1)[0]
        pso_optimize_predict = randn(1)[0]

        self.close_history.append(current_close)
        self.baseline_predict_history.append(baseline_predict)
        self.pso_optimize_predict_history.append(pso_optimize_predict)

        # baseline decision
        self.__baseline_decision(baseline_predict, current_close)

        # pso decision
        self.__pso_decision(pso_optimize_predict, current_close)

    def __buy(self, close, fund, stock):
        available_fund = fund / 3
        limit = int(np.round(available_fund / close) / 100) * 100
        if limit > 1:
            value = (limit * close) + self.TRADING_FEE
            return_fund = fund - value
            return_stock = stock + limit
            return_action = "buy"
            return return_fund, return_stock, return_action
        else:
            return self.__hold(fund=fund, stock=stock)

    def __sell(self, close, fund, stock):
        available_stock = stock / 3
        if available_stock >= 100:
            limit = int(available_stock / 100) * 100
            value = (limit * close) * (1 - self.TRADING_FEE)
            return_fund = fund + value
            return_stock = stock - limit
            return_action = "sell"
            return return_fund, return_stock, return_action
        else:
            return self.__hold(fund=fund, stock=stock)

    def __hold(self, fund, stock):
        return fund, stock, "hold"

    def __set_baseline_data(self, fund, stock, action):
        self.fund = fund
        self.stock = stock
        self.fund_history.append(self.fund)
        self.stock_history.append(self.stock)
        self.action_history.append(action)

    def __set_pso_data(self, fund, stock, action):
        self.pso_fund = fund
        self.pso_stock = stock
        self.pso_fund_history.append(self.pso_fund)
        self.pso_stock_history.append(self.pso_stock)
        self.pso_action_history.append(action)

    def __baseline_decision(self, predict, close):
        if predict > self.TRADING_CONSTANT:
            fund, stock, action = self.__buy(close=close, fund=self.fund, stock=self.stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)
        elif predict < -self.TRADING_CONSTANT:
            fund, stock, action = self.__sell(close=close, fund=self.fund, stock=self.stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)
        else:
            fund, stock, action = self.__hold(fund=self.fund, stock=self.stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)

    def __pso_decision(self, predict, close):
        if predict > self.TRADING_CONSTANT:
            fund, stock, action = self.__buy(close=close, fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)
        elif predict < -self.TRADING_CONSTANT:
            fund, stock, action = self.__sell(close=close, fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)
        else:
            fund, stock, action = self.__hold(fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)

    def start(self):
        for day in range(self.simulate_day):
            self.__trading(day)
        return 'start'

    def summary(self):
        return DataFrame(list(
            zip(self.baseline_predict_history, self.pso_optimize_predict_history, self.close_history, self.fund_history,
                self.pso_fund_history, self.stock_history, self.pso_stock_history, self.action_history,
                self.pso_action_history)),
            columns=['baseline_predict', 'pso_optimize_predict', 'close', 'fund', 'pso_fund', 'stock', 'pso_stock',
                     'action', 'pso_action'])
