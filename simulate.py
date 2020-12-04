from pandas import DataFrame


# import model


class Simulate:

    def __init__(self, simulate_day: int, simulate_data: DataFrame, initial_fund: int = 100000, trading_constant=0.5):
        self.simulate_day = simulate_day
        self.simulate_data = simulate_data
        self.fund = initial_fund
        self.TRADING_CONSTANT = trading_constant
        self.history = list()

    def __trading(self):
        return 'Hello World'
