# %%

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from indicator import Indicator
from preparation import Preparation
from ANN import ANN

# %%

# initial value
TRAIN_PATH = 'data/test_set/FB-18.csv'
TEST_PATH = 'data/test_set/FB-20.csv'

N_IN = 500  # number of date for training
N_OUT = 100  # number of date for predict
DIM = 6  # dimension for baseline model

EPOCH = 10
BATCH = 10

PARTICLE = 10  # number of PSO particle
ITERATION = 10  # number of PSO iteration
C1 = 1
C2 = 2
W = 0.01

# %%

# train data
read_data = pd.read_csv(TRAIN_PATH)
read_data['change'] = ((read_data['Close'] - read_data['Close'].shift(1)) / read_data['Close'].shift(1)) * 100
read_data.drop(['Adj Close'], axis=1, inplace=True)
read_data.drop([0], inplace=True)

# test data
test_data = pd.read_csv(TEST_PATH)
test_data['change'] = ((test_data['Close'] - test_data['Close'].shift(1)) / test_data['Close'].shift(1)) * 100
test_data.drop(['Adj Close'], axis=1, inplace=True)
test_data.drop([0], inplace=True)

# %%

train_X = read_data[['Open', 'Close', 'High', 'Low', 'Volume']]
train_y = read_data[['change']]

test_X = test_data[['Open', 'Close', 'High', 'Low', 'Volume']]
test_y = test_data[['change']]

# %%

# scale data
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0, 1))
# date_data_frame = read_data.loc[:, ['Date']]
# read_data.drop(['Date'], axis=1, inplace=True)
# training_set_scaled = sc.fit_transform(read_data)
# train_X = training_set_scaled[:-N_IN, :]
# train_y = training_set_scaled[-N_IN:, :]

# test_sc = MinMaxScaler(feature_range=(0, 1))
# test_date_data_frame = test_data.loc[:, ['Date']]
# test_data.drop(['Date'], axis=1, inplace=True)
# testing_set_scaled = test_sc.fit_transform(test_data)
# test_X = testing_set_scaled[:-N_OUT, :]
# test_y = testing_set_scaled[-N_OUT:, :]

# %%
base_line_model = Sequential()
base_line_model.add(Dense(N_IN, input_shape=(train_X.shape[1],), activation='relu'))
base_line_model.add(Dense(N_IN * 2, activation='relu'))
base_line_model.add(Dense(1, activation='linear'))
base_line_model.compile(optimizer='adam', loss='mean_squared_error')
history = base_line_model.fit(train_X, train_y, epochs=EPOCH, batch_size=BATCH, validation_split=0.2)

# %%
predict = base_line_model.predict(test_X)

# %%

plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['loss'], label='val_loss')
plt.legend(loc='upper left')
plt.show()
