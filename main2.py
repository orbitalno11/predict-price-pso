import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

from preparation import Preparation
from ANN import ANN
from indicator import Indicator
import  os
os.environ['CUDA_VISIBLE_DEVICE'] = '-1'

list_path = ['c','csco','cvx','dis','dis','fb','ko','mmm','msft','pfe','pg','v','wmt']
for i in list_path:
    # initial value
    TRAIN_PATH = 'data/test_set/'+i.upper()+'-20.csv'
    # PATH_ANN_MODEL = 'model/model-baseline-new/model-'+i+'18'

    PATH_PSO_MODEL = 'model/model-pso-new/model-pso-'+i+'20'

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
        (indicator_data['Close'] - indicator_data['ema_5_day']) / indicator_data['ema_5_day']) * 100
    data_set = indicator_data[['rsi', 'Histogram', 'Change of EMA', 'change']]
    ann.split_data_scale_transform(data_set)

    # # training baseline model
    # history = ann.baseline_train()
    # annModel = ann.get_baseline_model()
    # annModel.save(PATH_ANN_MODEL)

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

    # initialize swarm
    options = {'c1': C1, 'c2': C2, 'w': W}
    dimensions = 61
    max_bound = 10 * np.ones(dimensions)
    min_bound = -10 * np.ones(dimensions)
    bounds = (min_bound, max_bound)

    # define objective function for PSO
    # call instance of PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=PARTICLE, dimensions=dimensions, options=options, bounds=bounds)

    # optimize PSO
    cost, pos = optimizer.optimize(
        ann.objective_function, iters=ITERATION, verbose=1)

    # predict value from baseline model with PSO optimize
    pso_model = ann.model_pso(pos)
    # pso_predict = pso_model.predict(test_X)
    pso_model.save(PATH_PSO_MODEL)
