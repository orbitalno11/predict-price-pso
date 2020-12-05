import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from simulate import Simulator

data = pd.read_csv('data/test_set/FB-20.csv')

model1 = keras.models.load_model('model/model-baseline/model-fb')
model2 = keras.models.load_model('model/model-baseline/model-c')
simulator = Simulator(simulate_day=600, simulate_data=data,
                      baseline_model=model1, pso_model=model2)

simulator.start()
summary = simulator.summary()


plt.plot(summary['baseline_value_change'], label='baseline')
plt.plot(summary['pso_value_change'], label='pso')
plt.title('FB 2018')
plt.ylabel('change')
plt.xlabel('number of date')
plt.legend(loc='upper left')
plt.show()
