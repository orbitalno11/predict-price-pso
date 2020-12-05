from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from numpy.random import randn

from preparation import Preparation


# import model


class Simulator:

    def __init__(self, simulate_day: int, simulate_data: DataFrame, baseline_model, pso_model=None,
                 initial_fund: int = 1000000, trading_constant=0.5,
                 volume_constant=100, trading_fee=0.17):
        self.simulate_day = simulate_day
        self.simulate_data = simulate_data
        self.TRADING_CONSTANT = trading_constant
        self.VOLUME_CONSTANT = volume_constant
        self.TRADING_FEE = trading_fee / 100
        # initial model
        self.baseline_model = baseline_model
        self.pso_model = pso_model
        # initial fund
        self.first_day_fund = initial_fund
        self.baseline_fund = initial_fund
        self.pso_fund = initial_fund
        self.real_fund = initial_fund
        # initial stock
        self.baseline_stock = 0
        self.pso_stock = 0
        self.real_stock = 0
        # initial predict history
        self.baseline_predict_history = [0]
        self.pso_optimize_predict_history = [0]
        self.real_predict_history = [0]
        # initial fund history
        self.baseline_fund_history = [self.baseline_fund]
        self.pso_fund_history = [self.pso_fund]
        self.real_fund_history = [self.real_fund]
        # initial stock history
        self.baseline_stock_history = [self.baseline_stock]
        self.pso_stock_history = [self.pso_stock]
        self.real_stock_history = [self.real_stock]
        # initial action history
        self.baseline_action_history = ["start"]
        self.pso_action_history = ["start"]
        self.real_action_history = ["start"]
        # initial close history
        self.close_history = [np.nan]

    def __prepare_data(self):
        preparation = Preparation(self.simulate_data)
        preparation_data = preparation.calculate_per_change()
        sc = MinMaxScaler(feature_range=(0, 1))
        scale_data = sc.fit_transform(preparation_data)
        scale_data = scale_data[:, :5]
        self.preparation_data = preparation_data
        self.scale_data = scale_data
        self.sc = sc

    def __baseline_predict_data(self):
        baseline_predict = self.baseline_model.predict(self.scale_data)
        temp = np.concatenate((self.preparation_data.iloc[:, :5], baseline_predict), axis=1)
        reverse_data = self.sc.inverse_transform(temp)
        df = DataFrame(reverse_data, columns=[
            'Open', 'High', 'Low', 'Close', 'Volume', 'baseline_predict'])
        self.baseline_predict_data = df.copy()

    def __pso_predict_data(self):
        pso_predict = self.pso_model.predict(self.scale_data)
        temp = np.concatenate((self.preparation_data.iloc[:, :5], pso_predict), axis=1)
        reverse_data = self.sc.inverse_transform(temp)
        df = DataFrame(reverse_data, columns=[
            'Open', 'High', 'Low', 'Close', 'Volume', 'pso_predict_data'])
        self.pso_predict_data = df.copy()

    def __trading(self, day: int):
        current_close = self.preparation_data.loc[day:day, ['Close']].values[0][0]
        self.close_history.append(current_close)

        real_change = self.preparation_data.loc[day:day, ['change']].values[0][0]
        self.__real_decision(real_change, current_close)
        self.real_predict_history.append(real_change)

        # baseline decision
        baseline_predict = self.baseline_predict_data.loc[day:day, ['baseline_predict']].values[0][0]
        self.__baseline_decision(baseline_predict, current_close)
        self.baseline_predict_history.append(baseline_predict)

        # pso decision
        pso_optimize_predict = self.pso_predict_data.loc[day:day, ['pso_predict_data']].values[0][0] if self.pso_model is not None else 0
        self.__pso_decision(pso_optimize_predict, current_close)
        self.pso_optimize_predict_history.append(pso_optimize_predict)

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

    def __set_real_data(self, fund, stock, action):
        self.real_fund = fund
        self.real_stock = stock
        self.real_fund_history.append(fund)
        self.real_stock_history.append(stock)
        self.real_action_history.append(action)

    def __set_baseline_data(self, fund, stock, action):
        self.baseline_fund = fund
        self.baseline_stock = stock
        self.baseline_fund_history.append(fund)
        self.baseline_stock_history.append(stock)
        self.baseline_action_history.append(action)

    def __set_pso_data(self, fund, stock, action):
        self.pso_fund = fund
        self.pso_stock = stock
        self.pso_fund_history.append(fund)
        self.pso_stock_history.append(stock)
        self.pso_action_history.append(action)

    def __real_decision(self, predict, close):
        if predict > self.TRADING_CONSTANT:
            fund, stock, action = self.__buy(
                close=close, fund=self.real_fund, stock=self.real_stock)
            self.__set_real_data(fund=fund, stock=stock, action=action)
        elif predict < -self.TRADING_CONSTANT:
            fund, stock, action = self.__sell(
                close=close, fund=self.real_fund, stock=self.real_stock)
            self.__set_real_data(fund=fund, stock=stock, action=action)
        else:
            fund, stock, action = self.__hold(fund=self.real_fund, stock=self.real_stock)
            self.__set_real_data(fund=fund, stock=stock, action=action)

    def __baseline_decision(self, predict, close):
        if predict > self.TRADING_CONSTANT:
            fund, stock, action = self.__buy(
                close=close, fund=self.baseline_fund, stock=self.baseline_stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)
        elif predict < -self.TRADING_CONSTANT:
            fund, stock, action = self.__sell(
                close=close, fund=self.baseline_fund, stock=self.baseline_stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)
        else:
            fund, stock, action = self.__hold(fund=self.baseline_fund, stock=self.baseline_stock)
            self.__set_baseline_data(fund=fund, stock=stock, action=action)

    def __pso_decision(self, predict, close):
        if predict > self.TRADING_CONSTANT:
            fund, stock, action = self.__buy(
                close=close, fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)
        elif predict < -self.TRADING_CONSTANT:
            fund, stock, action = self.__sell(
                close=close, fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)
        else:
            fund, stock, action = self.__hold(
                fund=self.pso_fund, stock=self.pso_stock)
            self.__set_pso_data(fund=fund, stock=stock, action=action)

    def start(self):
        self.__prepare_data()
        self.__baseline_predict_data()
        if self.pso_model is not None:
            self.__pso_predict_data()
        for day in range(self.simulate_day):
            self.__trading(day)
        return 'start'

    def summary(self):
        df = DataFrame(list(
            zip(self.real_predict_history, self.baseline_predict_history, self.pso_optimize_predict_history,
                self.close_history, self.real_fund_history, self.baseline_fund_history, self.pso_fund_history,
                self.real_stock_history, self.baseline_stock_history, self.pso_stock_history,
                self.real_action_history, self.baseline_action_history, self.pso_action_history)),
            columns=['real_change', 'baseline_predict', 'pso_predict',
                     'close', 'real_fund', 'baseline_fund', 'pso_fund',
                     'real_stock', 'baseline_stock', 'pso_stock',
                     'real_action', 'baseline_action', 'pso_action'])
        df['real_value_change'] = (
                                              (df.real_fund - self.first_day_fund) / self.first_day_fund) * 100
        df['baseline_value_change'] = (
                                              (df.baseline_fund - self.first_day_fund) / self.first_day_fund) * 100
        df['pso_value_change'] = (
                                         (df.pso_fund - self.first_day_fund) / self.first_day_fund) * 100
        return df
