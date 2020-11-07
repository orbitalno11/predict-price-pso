# from indicator import Indicator
from preperation import Preperation
from model import LSTMNN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('data/AAPL-2014-2018.csv')
preperation = Preperation(df=dataset)
data = preperation.calculate_per_change()

model = LSTMNN(epochs=10, batch=10, n_in=30, n_out=3)
dataset = model.preprocess_data(data, data.columns)
train_X, train_y, test_X, test_y = model.spilt_data_scale_tranformt(dataset)
modelLSTM = model.baseline_model(train_X, train_y)
