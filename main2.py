import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

from preparation import Preparation
from ANN import ANN
from indicator import Indicator

# initial value
TRAIN_PATH = 'data/test_set/C-18.csv'
PATH_ANN_MODEL = 'model/model-baseline-new/model-c18/c'
# PATH_PSO_MODEL = 'model/model-pso-new/model-pso-c'

N_IN = 5  # number of date for training
N_OUT = 1  # number of date for predict

PARTICLE = 100  # number of PSO particle
ITERATION = 5  # number of PSO iteration
C1 = 2.5
C2 = 2.05
W = 0.7

# setup baseline model
ann = ANN(epochs=50, batch=13, n_in=N_IN, n_out=N_OUT)

# prepare train data
read_data = pd.read_csv(TRAIN_PATH)
preparation = Preparation(df=read_data)
data = preparation.calculate_per_change()

# create indicator
indicator = Indicator(data)
indicator_data = indicator.RSI()
indicator_data = indicator.EMA()
indicator_data = indicator.MACD()
indicator_data.dropna(inplace=True)

indicator_data['Change of EMA'] = (
    (indicator_data['ema_5_day'] - indicator_data['Close']) / indicator_data['ema_5_day']) * 100
data_set = indicator_data[['rsi', 'Histogram', 'Change of EMA', 'change']]
ann.split_data_scale_transform(data_set)

# training baseline model
history = ann.baseline_train()
annModel = ann.get_baseline_model()
# annModel.save(PATH_ANN_MODEL)
annModel.summary()

# # test baseline model
# ann.baseline_test()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'], loc='upper left')
# plt.show()

# # initialize swarm
# options = {'c1': C1, 'c2': C2, 'w': W}
# dimensions = 151
# max_bound = 5 * np.ones(dimensions)
# min_bound = -5 * np.ones(dimensions)
# bounds = (min_bound, max_bound)

# # define objective function for PSO
# # call instance of PSO
# optimizer = ps.single.GlobalBestPSO(
#     n_particles=PARTICLE, dimensions=dimensions, options=options, bounds=bounds)

# # optimize PSO
# cost, pos = optimizer.optimize(
#     ann.evaluate_model_pso, iters=ITERATION, verbose=1)

# # predict value from baseline model with PSO optimize
# pso_model = ann.evaluate_model_pso(pos)
# pso_predict = pso_model.predict(test_X)
# pso_model.save(PATH_PSO_MODEL)

# # # reversed value to original value
# # pso_temps = np.concatenate((test_X[0].reshape(1, DIM), pso_predict), axis=1)
# # pso_reversed = sc.inverse_transform(pso_temps)
# # pso_predict_changes = pso_reversed[:, -N_OUT:]

# # ann_predict_temps = np.concatenate((test_X[0].reshape(1, DIM), predict), axis=1)
# # ann_predict_reversed = sc.inverse_transform(ann_predict_temps)
# # ann_predict_changes = ann_predict_reversed[:, -N_OUT:]

# # # transform percent change to close value
# # close_change_ann = preparation_test.get_close_all(ann_predict_changes, last_close)
# # close_change_pso = preparation_test.get_close_all(pso_predict_changes, last_close)

# # # get close value
# # test_close_columns = test_data['Close']
# # close_change_columns_ann = close_change_ann['Close']
# # close_change_columns_pso = close_change_pso['Close']

# # # create close price dataframe
# # close_columns_frame_ann = [test_close_columns[:-N_OUT], close_change_columns_ann]
# # close_columns_ann = pd.concat(close_columns_frame_ann)
# # close_columns_ann = close_columns_ann.reset_index()
# # close_columns_ann.drop('index', axis=1, inplace=True)

# # close_columns_frame_pso = [test_close_columns[:-N_OUT], close_change_columns_pso]
# # close_columns_pso = pd.concat(close_columns_frame_pso)
# # close_columns_pso = close_columns_pso.reset_index()
# # close_columns_pso.drop('index', axis=1, inplace=True)

# # # create indicator
# # indicator_ann_predict = Indicator(close_columns_ann)
# # indicator_data_ann_predict = indicator_ann_predict.RSI()
# # indicator_data_ann_predict = indicator_ann_predict.EMA()
# # indicator_data_ann_predict = indicator_ann_predict.MACD()

# # indicator_pso_predict = Indicator(close_columns_pso)
# # indicator_data_pso_predict = indicator_pso_predict.RSI()
# # indicator_data_pso_predict = indicator_pso_predict.EMA()
# # indicator_data_pso_predict = indicator_pso_predict.MACD()

# # indicator_test = Indicator(test_data)
# # indicator_test_data = indicator_test.RSI()
# # indicator_test_data = indicator_test.EMA()
# # indicator_test_data = indicator_test.MACD()

# # show loss
# # plt.plot(history.history['loss'], label='ann_loss')
# # plt.plot(pso_model.history['loss'], label='pso_loss')
# # plt.title('Mean Square Error')
# # plt.ylabel('error')
# # plt.xlabel('epoch')
# # plt.legend(loc='upper left')
# # plt.show()

# # # show close value
# # plt.plot(predict, label='ANN')
# # plt.plot(pso_predict, label='PSO')
# # plt.plot(test_y, label='REAL VALUE')
# # plt.title('Close')
# # plt.ylabel('Close')
# # plt.xlabel('number of date')
# # plt.legend(loc='upper left')
# # plt.show()
