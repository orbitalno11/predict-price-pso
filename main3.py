import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from simulate import Simulator
from indicator import Indicator
N_DAY = 30

data = pd.read_csv('data/test_set/C-18.csv')

# create indicator
indicator = Indicator(data)
indicator_data = indicator.RSI()
indicator_data = indicator.EMA()
indicator_data = indicator.MACD()
indicator_data.dropna(inplace=True)
indicator_data['Change of EMA'] = (
        (indicator_data['Close'] - indicator_data['ema_5_day']) / indicator_data['ema_5_day']) * 100
data_set = indicator_data[['Close','rsi', 'Histogram', 'Change of EMA', 'change']]

model1 = keras.models.load_model('model/model-baseline-new/model-c18')
model2 = keras.models.load_model('model/model-pso-new/model-pso-c18')
simulator = Simulator(simulate_day=N_DAY, simulate_data=data_set,
                      baseline_model=model1, pso_model=model2)

simulator.start()
summary = simulator.summary()

plot_data = data.copy()
plot_data['change'] = (
                              (plot_data.Close - plot_data.Close.shift(1)) / plot_data.Close.shift(1)) * 100
plot_data = plot_data.loc[:N_DAY, ['change']]

day10 = summary.loc[10:10,
        ['buy_hold', 'real_value_change', 'baseline_value_change', 'pso_value_change', 'surplus_baseline',
         'surplus_pso']]

print('10-day')
for index in range(0, day10.columns.shape[0]):
    print('{} : {}'.format(day10.columns[index], day10.iloc[0, index]))

day30 = summary.loc[30:,
        ['buy_hold', 'real_value_change', 'baseline_value_change', 'pso_value_change', 'surplus_baseline',
         'surplus_pso']]

print('--------')
print('30-day')
for index in range(0, day30.columns.shape[0]):
    print('{} : {}'.format(day30.columns[index], day30.iloc[0, index]))

plt.plot(plot_data['change'], label='real')
plt.plot(summary['baseline_predict'], label='baseline')
plt.plot(summary['pso_predict'], label='pso')
plt.title('Percent change')
plt.ylabel('percent')
plt.xlabel('number of day')
plt.legend(loc='upper right')
plt.show()

plt.plot(summary['real_value_change'], label='real value')
plt.plot(summary['baseline_value_change'], label='baseline')
plt.plot(summary['pso_value_change'], label='pso')
plt.plot(summary['buy_hold'], label='b&h')
plt.title('2020')
plt.ylabel('change')
plt.xlabel('number of day')
plt.legend(loc='upper right')
plt.show()
