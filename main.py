from indicator import Indicator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/AAPL-2014-2018.csv')
indicator = Indicator(dataset=dataset, rsi_n_day=14)
rsi_data = indicator.RSI()

sns.lineplot(data=rsi_data[:30], x='Date', y='rsi')
plt.show()
