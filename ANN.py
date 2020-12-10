import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
        self.dim = 3

    # def set_train(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
    #     self.train['x'] = train_x.copy()
    #     self.train['y'] = train_y.copy()

    # def set_test(self, test_x: pd.DataFrame, test_y: pd.DataFrame):
    #     self.test['x'] = test_x
    #     self.test['y'] = test_y

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
        n_trian_precent = 0.8
        split = int(df.shape[0] * n_trian_precent)
        sc = MinMaxScaler(feature_range=(0, 1))
        data_scale = sc.fit_transform(df)
        data_trian, data_test = data_scale[:split, :], data_scale[split:, :]
        print(data_trian.shape)
        self.sc = sc
        self.train['x'], self.train['y'] = data_trian[:,
                                                      :-self.n_out], data_trian[:, -self.n_out:]
        self.test['x'], self.test['y'] = data_test[:,
                                                   :-self.n_out], data_trian[:-self.n_out:]

    def initial_baseline(self):
        self.base_line_model = Sequential()
        self.base_line_model.add(
            Dense(self.dim, input_dim=self.dim, activation='relu'))
        self.base_line_model.add(Dense(self.dim * 2, activation='relu'))
        self.base_line_model.add(Dense(self.dim, activation='relu'))
        self.base_line_model.add(Dense(self.n_out, activation='linear'))
        self.base_line_model.compile(
            optimizer='adam', loss='mean_squared_error')

    def baseline_train(self):
        self.initial_baseline()
        history = self.base_line_model.fit(self.train['x'], self.train['y'], epochs=self.epochs,
                                           batch_size=self.batch, validation_split=0.2)
        return history

    def baseline_test(self):
        self.predict_baseline = self.base_line_model.predict(self.test['x'])

    def reverse_test_baseline(self):
        data_scale = np.concatenate(
            (self.test['x'], self.predict_baseline), axis=1)
        self.predict_baseline = self.sc.inverse_transform(data_scale)
        return self.predict_baseline

    def get_baseline_model(self):
        return self.base_line_model

    def initial_model_pso(self, w: list):
        temp = self.dim * self.dim
        w1 = w[0:temp].reshape((self.dim, self.dim))
        b1 = w[temp:temp + self.dim].reshape((self.dim,))
        temp = temp + self.dim
        w2 = w[temp:temp + self.dim *
               (self.dim * 2)].reshape((self.dim, self.dim*2))
        temp = temp + self.dim * (self.dim * 2)
        b2 = w[temp:temp + self.dim * 2].reshape((self.dim*2,))
        temp = temp + self.dim * 2
        w3 = w[temp:temp + (self.dim * self.dim * 2)
               ].reshape((self.dim*2, self.dim))
        temp = temp + (self.dim * self.dim * 2)
        b3 = w[temp:temp + self.dim].reshape((self.dim,))
        temp = temp + self.dim
        w4 = w[temp:temp + self.dim].reshape((self.dim, 1))
        temp = temp + self.dim
        b4 = w[temp:temp + 1]
        weight = [w1, b1, w2, b2, w3, b3, w4, b4]

        self.pso_model = Sequential()
        self.pso_model.add(
            Dense(self.dim, input_dim=self.dim, activation='relu'))
        self.pso_model.add(Dense(self.dim * 2, activation='relu'))
        self.pso_model.add(Dense(self.dim, activation='relu'))
        self.pso_model.add(Dense(self.n_out, activation='linear'))
        self.pso_model.set_weights(weight)
        self.pso_model.compile(optimizer='adam', loss='mean_squared_error')

    def evaluate_model_pso(self, w: list):
        self.initial_model_pso(w)
        # self.pso_model.fit(self.train['x'], self.train['y'], epochs=0, batch_size=self.batch)
        evaluate = self.pso_model.evaluate(
            self.train['x'], self.train['y'], batch_size=self.batch)
        return evaluate

    def objective_function(self, x):
        n_particles = x.shape[0]
        loss = [self.evaluate_model_pso(x[i]) for i in range(n_particles)]
        return np.array(loss)

    def model_pso(self, w: list):
        self.initial_model_pso(w)
        return self.pso_model

    def model_pso_test(self):
        self.predict_pso = self.pso_model.predict(self.test['x'])

    def reverse_test_pso(self):
        data_scale = np.concatenate(
            (self.test['x'], self.predict_pso), axis=1)
        self.predict_pso = self.sc.inverse_transform(data_scale)
        return self.predict_pso
