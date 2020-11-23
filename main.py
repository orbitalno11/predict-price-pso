from indicator import Indicator
from preparation import Preparation
from model import RnnNN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import data train | stock apple
dataset = pd.read_csv('data/AAPL-2014-2018.csv')
# preparation train data
preparation = Preparation(df=dataset)
data = preparation.calculate_per_change()
n_in = 30
n_out = 10
model = RnnNN(epochs=10, batch=10, n_in=n_in, n_out=n_out)
dataset = model.preprocess_data(data, data.columns)
# split and scale transform training data
train_X, train_y, sc = model.split_data_scale_tranformt(dataset)
# train model RNN
modelRNN, history = model.baseline_model(train_X, train_y)

# import data test | stock apple
test_data = pd.read_csv('data/AAPLv.2.csv')
# preparation test data
preparation_test = Preparation(df=test_data)
# get close last day
last_close = preparation_test.get_close()
data_test = preparation_test.calculate_per_change()
# split and scale transform test data
test_sample = model.preprocess_data(data_test.values, data_test.columns)
test_transform = sc.transform(test_sample)
test_X, test_y = test_transform[:, :-n_out], test_transform[:, -n_out:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# predict
predict = modelRNN.predict(test_X)

# # inverse data
reshapeTest = test_X.reshape((test_X.shape[0], test_X.shape[2]))
reshapePredict = predict.reshape((predict.shape[0], predict.shape[2]))
temp = np.concatenate((reshapeTest, reshapePredict), axis=1)
reverse = sc.inverse_transform(temp)
predict_change = reverse[:, -n_out:]

# change percent change to close
close_change = preparation_test.get_close_all(predict_change, last_close)

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

# plot compare real and predict percentchage
plt.plot(test_y[0], label='real_change')
plt.plot(predict_change[0], label='predict_change')
plt.legend()
plt.show()

# get Close
test_close_columns = test_data['Close']
close_change_columns = close_change['Close']

# create close price dataframe
close_columns_frame = [test_close_columns, close_change_columns]
close_columns = pd.concat(close_columns_frame)
close_columns = close_columns.reset_index()
close_columns.drop('index', axis=1, inplace=True)

indicator = Indicator(close_columns)
indicator_data = indicator.RSI()
indicator_data = indicator.EMA()
indicator_data = indicator.MACD()
