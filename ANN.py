import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


class ANN:

    def __init__(self, epochs: int, batch: int, n_in: int, n_out: int):
        self.epochs = epochs
        self.batch = batch
        self.n_in = n_in
        self.n_out = n_out
        self.train = dict()
        self.test = dict()
        self.base_line_model = Sequential()
        self.pso_model = Sequential()
        self.dim = 5

    def set_train(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        self.train['x'] = train_x.copy()
        self.train['y'] = train_y.copy()

    def set_test(self, test_x: pd.DataFrame, test_y: pd.DataFrame):
        self.test['x'] = test_x
        self.test['y'] = test_y

    # def pre_process_data(self, df: pd.DataFrame, feature_name: list, drop_nan=True) -> np.ndarray:
    #     n_col = 1 if type(df) is list else df.shape[1]
    #     dataset = pd.DataFrame(df)
    #     cols, names, d_col = list(), list(), list()

    #     # input sequence (t-n, ... t-1)
    #     for i in range(self.n_in, 0, -1):
    #         cols.append(dataset.shift(i))
    #         names += [f'{feature_name[j]}(t-{i})' for j in range(n_col)]

    #     # forecast sequence (t, t+1, ... t+n)
    #     for i in range(0, self.n_out):
    #         cols.append(dataset.shift(-i))
    #         if i == 0:
    #             temp = [f'{feature_name[j]}(t)' for j in range(n_col)]
    #             names += temp
    #             d_col += temp
    #         else:
    #             temp = [f'{feature_name[j]}(t+{i})' for j in range(n_col)]
    #             names += temp
    #             d_col += temp
    #     d_col = [val for i, val in enumerate(
    #         d_col) if not val.startswith('change')]
    #     # put it all together
    #     agg = pd.concat(cols, axis=1)
    #     agg.columns = names
    #     if drop_nan:
    #         agg.dropna(inplace=True)
    #     agg.drop(d_col, axis=1, inplace=True)
    #     return agg

    def split_data_scale_transform(self, df: pd.DataFrame) -> tuple:
        # Feature Scaling
        df = df.to_numpy()
        df_x , df_y = df[:,:-self.n_out], df[:,-self.n_out:]
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(df)
        train_X, train_y = training_set_scaled[:, :-self.n_out], training_set_scaled[:, -self.n_out:]
        return train_X, train_y , sc, df_x, df_y

    def initial_baseline(self):
        self.base_line_model = Sequential()
        self.base_line_model.add(Dense(self.dim, input_dim=self.dim, activation='relu'))
        self.base_line_model.add(Dense(self.dim * 2, activation='relu'))
        self.base_line_model.add(Dense(self.n_out, activation='sigmoid'))
        self.base_line_model.compile(optimizer='adam', loss='mean_squared_error')

    def baseline_train(self):
        self.initial_baseline()
        history = self.base_line_model.fit(self.train['x'], self.train['y'], epochs=self.epochs,
                                           batch_size=self.batch, validation_split=0.2)
        return history

    def get_baseline_model(self):
        return self.base_line_model

    def initial_model_pso(self, w: list):
        temp = self.dim * self.n_in
        w1 = w[0:temp].reshape((self.dim, self.n_in))
        b1 = w[temp:temp + self.n_in].reshape((self.n_in,))
        temp = temp + self.n_in
        w2 = w[temp:temp + (self.n_in * self.n_in * 2)].reshape((self.n_in, self.n_in * 2))
        temp = temp + (self.n_in * self.n_in * 2)
        b2 = w[temp:temp + (self.n_in * 2)].reshape((self.n_in * 2,))
        temp = temp + (self.n_in * 2)
        w3 = w[temp:temp + (self.n_in * 2 * self.n_out)].reshape((self.n_in * 2, self.n_out))
        temp = temp + (self.n_in * 2 * self.n_out)
        b3 = w[temp:temp + self.n_out].reshape((self.n_out,))
        weight = [w1, b1, w2, b2, w3, b3]

        self.pso_model = Sequential()
        self.pso_model.add(Dense(self.n_in, input_dim=self.dim, activation='relu'))
        self.pso_model.add(Dense(self.n_in * 2, activation='relu'))
        self.pso_model.add(Dense(self.n_out, activation='sigmoid'))
        self.pso_model.set_weights(weight)
        self.pso_model.compile(optimizer='adam', loss='mean_squared_error')

    def evaluate_model_pso(self, w: list):
        self.initial_model_pso(w)

        # self.pso_model.fit(self.train['x'], self.train['y'], epochs=0, batch_size=self.batch)
        evaluate = self.pso_model.evaluate(self.train['x'], self.train['y'], batch_size=self.batch)
        return evaluate

    def model_pso(self, w: list):
        self.initial_model_pso(w)
        return self.pso_model
