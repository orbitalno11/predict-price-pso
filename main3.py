import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from simulate import Simulator

data = pd.read_csv('data/test_set/FB-20.csv')

model1 = keras.models.load_model('model/model-baseline/model-fb')
model2 = keras.models.load_model('model/model-pso/model-pso-fb')
simulator = Simulator(simulate_day=120, simulate_data=data,
                      baseline_model=model1, pso_model=model2, trading_constant=0.4)

simulator.start()
summary = simulator.summary()

plot_data = data.copy()
plot_data['change'] = ((plot_data.Close - plot_data.Close.shift(1)) / plot_data.Close.shift(1)) * 100
plot_data = plot_data.loc[:120, ['change']]

plt.plot(plot_data['change'], label='real')
plt.plot(summary['baseline_predict'], label='baseline')
plt.plot(summary['pso_predict'], label='pso')
plt.title('Percent change')
plt.ylabel('percent')
plt.xlabel('number of date')
plt.legend(loc='upper right')
plt.show()

plt.plot(summary['real_value_change'], label='real value')
plt.plot(summary['baseline_value_change'], label='baseline')
plt.plot(summary['pso_value_change'], label='pso')
plt.title('2020')
plt.ylabel('change')
plt.xlabel('number of date')
plt.legend(loc='upper right')
plt.show()
