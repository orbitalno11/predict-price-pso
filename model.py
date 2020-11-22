import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


class RnnNN:

    def __init__(self, epochs: int, batch: int, n_in: int, n_out: int):
        self.epochs = epochs
        self.batch = batch
        self.n_in = n_in
        self.n_out = n_out

    def preprocess_data(self, df: pd.DataFrame, feature_name: list, dropnan=True) -> np.ndarray:
        n_col = 1 if type(df) is list else df.shape[1]
        dataset = pd.DataFrame(df)
        cols, names, d_col = list(), list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(dataset.shift(i))
            names += [f'{feature_name[j]}(t-{i})' for j in range(n_col)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(dataset.shift(-i))
            if i == 0:
                temp = [f'{feature_name[j]}(t)' for j in range(n_col)]
                names += temp
                d_col += temp
            else:
                temp = [f'{feature_name[j]}(t+{i})' for j in range(n_col)]
                names += temp
                d_col += temp
        d_col = [val for i, val in enumerate(
            d_col) if not val.startswith('change')]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        agg.drop(d_col, axis=1, inplace=True)
        return agg

    def split_data_scale_tranformt(self, df: pd.DataFrame) -> tuple:
        # Feature Scaling
        df = df.to_numpy()
        # split = int(df.shape[0]*0.9)
        # train = df[:split, :]
        # test = df[split:, :]
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(df)
        train_X, train_y = training_set_scaled[:, :-
                                               self.n_out], training_set_scaled[:, -self.n_out:]
        # test_X, test_y = test[:, :-self.n_out], test[:, -self.n_out:]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        return train_X, train_y, sc

    def baseline_model(self, train_X, train_y):
        model = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        model.add(SimpleRNN(self.n_in, return_sequences=True,
                            input_shape=(train_X.shape[1], train_X.shape[2])))
        # model.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        model.add(SimpleRNN(60, return_sequences=True))
        # model.add(Dropout(0.2))
        # Adding a third LSTM layer and some Dropout regularisation
        model.add(SimpleRNN(60, return_sequences=True))
        # model.add(Dropout(0.2))
        # Adding the output layer
        model.add(Dense(units=self.n_out))
        # Compiling the RNN
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Fitting the RNN to the Training set
        history = model.fit(train_X, train_y, epochs=self.epochs,
                            batch_size=self.batch, validation_split=0.2)
        return model, history

    # def pso_lstm_nn(self, train_X, train_y):
    #     model = Sequential()
    #     # Adding the first LSTM layer and some Dropout regularisation
    #     model.add(LSTM(units=60, return_sequences=True,
    #                    input_shape=(train_X.shape[1], train_X.shape[2])))
    #     # model.add(Dropout(0.2))
    #     # Adding a second LSTM layer and some Dropout regularisation
    #     model.add(LSTM(units=60, return_sequences=True))
    #     # model.add(Dropout(0.2))

    #     # Adding a third LSTM layer and some Dropout regularisation
    #     model.add(LSTM(units=60, return_sequences=True))
    #     # model.add(Dropout(0.2))
    #     # Adding the output layer
    #     model.add(Dense(units=self.n_out))
    #     # Compiling the RNN
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     # Fitting the RNN to the Training set
    #     history = model.fit(train_X, train_y, epochs=self.epochs,
    #                         batch_size=self.batch, validation_split=0.2)
    #     return model, history
