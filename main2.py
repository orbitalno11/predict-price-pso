import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from indicator import Indicator

from preparation import Preparation
from ANN import ANN

read_data = pd.read_csv('data/AAPL-2014-2018.csv')
preparation = Preparation(df=read_data)
data = preparation.calculate_per_change()
n_in = 14
n_out = 10
dim = n_in * 6
ann = ANN(epochs=10, batch=10, n_in=n_in, n_out=n_out)
dataset = ann.pre_process_data(data, data.columns)
train_X, train_y, sc = ann.split_data_scale_transform(dataset)
ann.set_train(train_X, train_y)

history = ann.baseline_train()
annModel = ann.get_baseline_model()

test_data = pd.read_csv('data/AAPL-30.csv')
preparation_test = Preparation(df=test_data)
last_close = preparation_test.get_close()
data_test = preparation_test.calculate_per_change()

test_sample = ann.pre_process_data(data_test, data_test.columns)
test_transform = sc.transform(test_sample)
test_X, test_y = test_transform[:, :-n_out], test_transform[:, -n_out:]
ann.set_test(test_X[0].reshape(1,dim), test_y[0])

predict = annModel.predict(test_X[0].reshape(1,dim))
weight = annModel.get_weights()
# for w in weight :
#     print(np.amax(w),np.amin(w))
#     print('------------')
# annModel.summary()

plt.plot(test_y[0], label='real_change')
plt.plot(predict[0], label='predict_change')
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# annModel.summary()

import pyswarms as ps

# Initialize swarm
options = {'c1': 1, 'c2': 1, 'w': 0.01}
dimensions = annModel.count_params()
max_bound = 0.5 * np.ones(dimensions)
min_bound = - max_bound
bounds = (min_bound, max_bound)

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)


def objective_function(x):
    n_particles = x.shape[0]
    j = [ann.evaluate_model_pso(x[i]) for i in range(n_particles)]
    return np.array(j)


cost, pos = optimizer.optimize(objective_function, iters=10, verbose=1)

pso_model = ann.model_pso(pos)
pso_predict = pso_model.predict(test_X[0].reshape(1,dim))

plt.plot(test_y[0], label='real_change')
plt.plot(predict[0], label='ann_predict_change')
plt.plot(pso_predict[0], label='pso_predict_change')
plt.legend()
plt.show()

# reversed
pso_temps = np.concatenate((test_X[0].reshape(1,dim), pso_predict), axis=1)
pso_reversed = sc.inverse_transform(pso_temps)
pso_predict_changes = pso_reversed[:, -n_out:]

predict_temps = np.concatenate((test_X[0].reshape(1,dim), predict), axis=1)
predict_reversed = sc.inverse_transform(predict_temps)
predict_changes = predict_reversed[:, -n_out:]

# temps = np.concatenate((test_X[0].reshape(1,dim), test_y), axis=1)
# reversed = sc.inverse_transform(temps)
# changes = reversed[:, -n_out:]

# change percent change to close
close_change_predict = preparation_test.get_close_all(predict_changes, last_close)
close_change_pso = preparation_test.get_close_all(pso_predict_changes, last_close)

# get Close
test_close_columns = test_data['Close']
close_change_columns_predict = close_change_predict['Close']
close_change_columns_pso = close_change_pso['Close']

# create close price dataframe
close_columns_frame_predict = [test_close_columns[:-n_out], close_change_columns_predict]
close_columns_predict = pd.concat(close_columns_frame_predict)
close_columns_predict = close_columns_predict.reset_index()
close_columns_predict.drop('index', axis=1, inplace=True)

close_columns_frame_pso = [test_close_columns[:-n_out], close_change_columns_pso]
close_columns_pso = pd.concat(close_columns_frame_pso)
close_columns_pso = close_columns_pso.reset_index()
close_columns_pso.drop('index', axis=1, inplace=True)

indicator_predict = Indicator(close_columns_predict)
indicator_data_predict = indicator_predict.RSI()
indicator_data_predict = indicator_predict.EMA()
indicator_data_predict = indicator_predict.MACD()

indicator_pso = Indicator(close_columns_pso)
indicator_data_pso = indicator_pso.RSI()
indicator_data_pso = indicator_pso.EMA()
indicator_data_pso = indicator_pso.MACD()

indicator = Indicator(test_data)
indicator_data = indicator.RSI()
indicator_data = indicator.EMA()
indicator_data = indicator.MACD()

plt.plot(indicator_data_predict['rsi'])
plt.plot(indicator_data_pso['rsi'])
plt.plot(indicator_data['rsi'])
plt.title('RSI')
plt.ylabel('RSI')
plt.xlabel('number of date')
plt.legend(['ANN', 'PSO', 'real value'], loc='upper left')
plt.show()

plt.plot(indicator_data_predict['ema_5_day'])
plt.plot(indicator_data_pso['ema_5_day'])
plt.plot(indicator_data['ema_5_day'])
plt.title('EMA 5 day')
plt.ylabel('EMA')
plt.xlabel('number of date')
plt.legend(['ANN', 'PSO'], loc='upper left')
plt.show()

plt.plot(indicator_data_predict['ema_12_day'])
plt.plot(indicator_data_pso['ema_12_day'])
plt.plot(indicator_data['ema_12_day'])
plt.title('EMA 12 day')
plt.ylabel('EMA')
plt.xlabel('number of date')
plt.legend(['ANN', 'PSO', 'real value'], loc='upper left')
plt.show()

plt.plot(indicator_data_predict['MACD'])
plt.plot(indicator_data_pso['MACD'])
plt.plot(indicator_data['MACD'])
plt.title('MACD')
plt.ylabel('MACD')
plt.xlabel('number of date')
plt.legend(['ANN', 'PSO', 'real value'], loc='upper left')
plt.show()