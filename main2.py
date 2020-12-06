import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

# from indicator import Indicator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from preparation import Preparation
from ANN import ANN

# initial value
TRAIN_PATH = 'data/test_set/CSCO-18.csv'
PATH_ANN_MODEL = 'model/model-baseline/model-cs'
PATH_PSO_MODEL = 'model/model-pso/model-pso-csco'

N_IN = 5  # number of date for training
N_OUT = 1  # number of date for predict
DIM = N_IN * 6  # dimension for baseline model

PARTICLE = 100  # number of PSO particle
ITERATION = 5  # number of PSO iteration
C1 = 2.5
C2 = 2.05
W = 0.7

# setup baseline model
ann = ANN(epochs=200, batch=3, n_in=N_IN, n_out=N_OUT)

# prepare train data
read_data = pd.read_csv(TRAIN_PATH)
preparation = Preparation(df=read_data)
data = preparation.calculate_per_change()
# dataset = ann.pre_process_data(data, data.columns)
train_X, train_y, sc = ann.split_data_scale_transform(data)
ann.set_train(train_X, train_y)

# # training baseline model
# history = ann.baseline_train()
# annModel = ann.get_baseline_model()
# annModel.save(PATH_ANN_MODEL)
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # prepare test data
# test_data = pd.read_csv(TEST_PATH)
# preparation_test = Preparation(df=test_data)
# # last_close = preparation_test.get_close()
# data_test = preparation_test.calculate_per_change()

# # test_sample = ann.pre_process_data(data_test, data_test.columns)
# test_transform = sc.transform(data_test)
# test_X, test_y = test_transform[:, :-N_OUT], test_transform[:, -N_OUT:]
# # ann.set_test(test_X[0].reshape(1, DIM), test_y[0])

# # predict value from baseline model
# predict = annModel.predict(test_X)
# weight = annModel.get_weights()
# for w in weight:
#     print(np.amax(w), np.amin(w))
#     print('-------------------------------')

# initialize swarm
options = {'c1': C1, 'c2': C2, 'w': W}
dimensions = 151
max_bound = 3 * np.ones(dimensions)
min_bound = -2 * np.ones(dimensions)
bounds = (min_bound, max_bound)

# define objective function for PSO
def evaluate_model_pso( w: list):
    w1 = w[0:25].reshape((5, 5))
    b1 = w[25:30].reshape((5,))
    w2 = w[30:80].reshape((5, 10))
    b2 = w[80:90].reshape((10,))
    w3 = w[90:140].reshape((10, 5))
    b3 = w[140:145].reshape((5,))
    w4 = w[145:150].reshape((5, 1))
    b4 = w[150:151]
    weight = [w1, b1, w2, b2, w3, b3, w4, b4]

    pso_model = Sequential()
    pso_model.add(Dense(5, input_dim=5, activation='relu'))
    pso_model.add(Dense(5 * 2, activation='relu'))
    pso_model.add(Dense(5 ,activation='relu'))
    pso_model.add(Dense(1, activation='linear'))
    pso_model.set_weights(weight)
    pso_model.compile(optimizer='adam', loss='mean_squared_error')

    # self.pso_model.fit(self.train['x'], self.train['y'], epochs=0, batch_size=self.batch)
    evaluate = pso_model.evaluate(train_X, train_y, batch_size=3)
    return evaluate

def objective_function(x):
    n_particles = x.shape[0]
    j = [evaluate_model_pso(x[i]) for i in range(n_particles)]
    return np.array(j)

def evaluate_model_pso_model( w: list):
    w1 = w[0:25].reshape((5, 5))
    b1 = w[25:30].reshape((5,))
    w2 = w[30:80].reshape((5, 10))
    b2 = w[80:90].reshape((10,))
    w3 = w[90:140].reshape((10, 5))
    b3 = w[140:145].reshape((5,))
    w4 = w[145:150].reshape((5, 1))
    b4 = w[150:151]
    weight = [w1, b1, w2, b2, w3, b3, w4, b4]

    pso_model = Sequential()
    pso_model.add(Dense(5, input_dim=5, activation='relu'))
    pso_model.add(Dense(5 * 2, activation='relu'))
    pso_model.add(Dense(5 ,activation='relu'))
    pso_model.add(Dense(1, activation='linear'))
    pso_model.set_weights(weight)
    pso_model.compile(optimizer='adam', loss='mean_squared_error')
    return pso_model

# call instance of PSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=PARTICLE, dimensions=dimensions, options=options)

# optimize PSO
cost, pos = optimizer.optimize(objective_function, iters=ITERATION, verbose=1)

# predict value from baseline model with PSO optimize
pso_model = evaluate_model_pso_model(pos)
# pso_predict = pso_model.predict(test_X)
# pso_model.save(PATH_PSO_MODEL)
# # reversed value to original value
# pso_temps = np.concatenate((test_X[0].reshape(1, DIM), pso_predict), axis=1)
# pso_reversed = sc.inverse_transform(pso_temps)
# pso_predict_changes = pso_reversed[:, -N_OUT:]

# ann_predict_temps = np.concatenate((test_X[0].reshape(1, DIM), predict), axis=1)
# ann_predict_reversed = sc.inverse_transform(ann_predict_temps)
# ann_predict_changes = ann_predict_reversed[:, -N_OUT:]

# # transform percent change to close value
# close_change_ann = preparation_test.get_close_all(ann_predict_changes, last_close)
# close_change_pso = preparation_test.get_close_all(pso_predict_changes, last_close)

# # get close value
# test_close_columns = test_data['Close']
# close_change_columns_ann = close_change_ann['Close']
# close_change_columns_pso = close_change_pso['Close']

# # create close price dataframe
# close_columns_frame_ann = [test_close_columns[:-N_OUT], close_change_columns_ann]
# close_columns_ann = pd.concat(close_columns_frame_ann)
# close_columns_ann = close_columns_ann.reset_index()
# close_columns_ann.drop('index', axis=1, inplace=True)

# close_columns_frame_pso = [test_close_columns[:-N_OUT], close_change_columns_pso]
# close_columns_pso = pd.concat(close_columns_frame_pso)
# close_columns_pso = close_columns_pso.reset_index()
# close_columns_pso.drop('index', axis=1, inplace=True)

# # create indicator
# indicator_ann_predict = Indicator(close_columns_ann)
# indicator_data_ann_predict = indicator_ann_predict.RSI()
# indicator_data_ann_predict = indicator_ann_predict.EMA()
# indicator_data_ann_predict = indicator_ann_predict.MACD()

# indicator_pso_predict = Indicator(close_columns_pso)
# indicator_data_pso_predict = indicator_pso_predict.RSI()
# indicator_data_pso_predict = indicator_pso_predict.EMA()
# indicator_data_pso_predict = indicator_pso_predict.MACD()

# indicator_test = Indicator(test_data)
# indicator_test_data = indicator_test.RSI()
# indicator_test_data = indicator_test.EMA()
# indicator_test_data = indicator_test.MACD()

# show loss
# plt.plot(history.history['loss'], label='ann_loss')
# plt.plot(pso_model.history['loss'], label='pso_loss')
# plt.title('Mean Square Error')
# plt.ylabel('error')
# plt.xlabel('epoch')
# plt.legend(loc='upper left')
# plt.show()

# # show close value
# plt.plot(predict, label='ANN')
# plt.plot(pso_predict, label='PSO')
# plt.plot(test_y, label='REAL VALUE')
# plt.title('Close')
# plt.ylabel('Close')
# plt.xlabel('number of date')
# plt.legend(loc='upper left')
# plt.show()

# # show RSI 14
# plt.plot(indicator_data_ann_predict['rsi'], label='ANN')
# plt.plot(indicator_data_pso_predict['rsi'], label='PSO')
# plt.plot(indicator_test_data['rsi'], label='REAL VALUE')
# plt.title('RSI')
# plt.ylabel('RSI')
# plt.xlabel('number of date')
# plt.legend(loc='upper left')
# plt.show()

# # show EMA 5 day
# plt.plot(indicator_data_ann_predict['ema_5_day'], label='ANN')
# plt.plot(indicator_data_pso_predict['ema_5_day'], label='PSO')
# plt.plot(indicator_test_data['ema_5_day'], label='REAL VALUE')
# plt.title('EMA 5 day')
# plt.ylabel('EMA')
# plt.xlabel('number of date')
# plt.legend(loc='upper left')
# plt.show()

# # show EMA 12 day
# plt.plot(indicator_data_ann_predict['ema_12_day'], label='ANN')
# plt.plot(indicator_data_pso_predict['ema_12_day'], label='PSO')
# plt.plot(indicator_test_data['ema_12_day'], label='REAL VALUE')
# plt.title('EMA 12 day')
# plt.ylabel('EMA')
# plt.xlabel('number of date')
# plt.legend(loc='upper left')
# plt.show()

# # show MACD
# plt.plot(indicator_data_ann_predict['MACD'], label='ANN')
# plt.plot(indicator_data_pso_predict['MACD'], label='PSO')
# plt.plot(indicator_test_data['MACD'], label='REAL VALUE')
# plt.title('MACD')
# plt.ylabel('MACD')
# plt.xlabel('number of date')
# plt.legend(loc='upper left')
# plt.show()
