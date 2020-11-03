import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import matplotlib.pyplot as plt

dataset = pd.read_csv('data/AAPL-2014-2018.csv')

X = dataset[['Open', 'High', 'Low']]
Y = dataset[['Close']]

X_train = X[0:1000]
Y_train = Y[0:1000]

X_test = X[1000:]
Y_test = Y[1000:]

X_test.reset_index(inplace=True, drop=True)
Y_test.reset_index(inplace=True, drop=True)


def baseline_model():
    model = Sequential()
    model.add(Dense(4, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5)
estimator.fit(X_train, Y_train)
y_pred = estimator.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(mse)

plt.plot(Y_test, label='y_test')
plt.plot(y_pred, label='y_pred')
plt.legend()
plt.show()

