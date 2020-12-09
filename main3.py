import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from simulate import Simulator

N_DAY = 30

data = pd.read_csv('data/test_set/DIS-20.csv')
# data = pd.read_csv('data/FB.csv')

model1 = keras.models.load_model('model/model-baseline/model-dis')
model2 = keras.models.load_model('model/model-pso/model-pso-dis')
simulator = Simulator(simulate_day=N_DAY, simulate_data=data,
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
