from indicator import Indicator
from preperation import Preperation
from model import LSTMNN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# dataset = pd.read_csv('data/AAPLv.2.csv')
dataset = pd.read_csv('data/AAPL-2014-2018.csv')
preperation = Preperation(df=dataset)
data = preperation.calculate_per_change()
n_in = 20
n_out = 10
model = LSTMNN(epochs=10, batch=10, n_in=n_in, n_out=n_out)
dataset = model.preprocess_data(data, data.columns)
train_X, train_y, test, sc = model.spilt_data_scale_tranformt(dataset)
modelLSTM, history = model.baseline_model(train_X, train_y)
# eva = modelLSTM.evaluate(test_X, test_y)
test_set_scaled = sc.transform(test)
test_X, test_y = test_set_scaled[:, :-n_out], test_set_scaled[:, -n_out:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
predict = modelLSTM.predict(test_X)

# inverse data
reshapeTest = test_X.reshape((test_X.shape[0], test_X.shape[2]))
reshapePredict = predict.reshape((predict.shape[0], predict.shape[2]))
temp = np.concatenate((reshapeTest, reshapePredict), axis=1)
reverse = sc.inverse_transform(temp)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# sc.transform(test_X[:,0,:])
# _ = predict[:,0,:]
# temp = np.concatenate((train_X, test_X), axis=0)
# test = sc.inverse_transform(temp[:, 0, :])

# reverse = predict[0, 0, :]
plt.plot(test_y[0], label='real_price')
# pred = reverse[:, -30:]
plt.plot(predict[0, 0, :], label='predict')
plt.legend()
plt.show()

# last_day_close =
new_df = pd.DataFrame()
# indicator = Indicator()

for i in range (n_out):
    test_X[120]

test_X[0,0,119]
reverse[0,]
_ = sc.inverse_transform(test)
print(_[0][128])
