import os
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

from simulate import Simulator
from indicator import Indicator

N_DAY = 30
N_TRAIN = 0.8

list_path = os.listdir('input')
for name in list_path:
    FILE_PATH = 'input/{}'.format(name)
    temp = name.split('-')
    temp_str = temp[0] + temp[1]
    model_name = temp_str.lower().split('.')[0]
    BASELINE_MODEL_PATH = 'model/model-baseline-new/model-{}'.format(model_name)
    PSO_MODEL_PATH = 'model/model-pso-new/model-pso-{}'.format(model_name)

    data = pd.read_csv(FILE_PATH)
    ann = keras.models.load_model(BASELINE_MODEL_PATH)
    pso = keras.models.load_model(PSO_MODEL_PATH)

    # create indicator
    indicator = Indicator(data)
    indicator_data = indicator.RSI()
    indicator_data = indicator.EMA()
    indicator_data = indicator.MACD()
    indicator_data.dropna(inplace=True)
    indicator_data['change_of_ema'] = ((indicator_data['Close'] - indicator_data['ema_5_day']) / indicator_data['ema_5_day']) * 100
    data_set = indicator_data[['Close', 'rsi', 'Histogram', 'change_of_ema', 'change']]

    simulator = Simulator(simulate_day=N_DAY, simulate_data=data_set, baseline_model=ann, pso_model=pso)
    simulator.start()
    summary = simulator.summary()

    split = int(data_set.shape[0] * N_TRAIN)
    plot_data = data_set.iloc[split:, :]
    plot_data.reset_index(drop=True, inplace=True)
    plot_data = plot_data.loc[:N_DAY, ['change']]

    print('Summary of {}'.format(name.split('.')[0]))
    day10 = summary.loc[10:10,
            ['buy_hold', 'real_value_change', 'baseline_value_change', 'pso_value_change', 'surplus_baseline',
             'surplus_pso']]

    print('10-day')
    for index in range(0, day10.columns.shape[0]):
        print('{} : {}'.format(day10.columns[index], day10.iloc[0, index]))

    day30 = summary.loc[30:,
            ['buy_hold', 'real_value_change', 'baseline_value_change', 'pso_value_change', 'surplus_baseline',
             'surplus_pso']]

    print('--------')
    print('30-day')
    for index in range(0, day30.columns.shape[0]):
        print('{} : {}'.format(day30.columns[index], day30.iloc[0, index]))
    print('######################################')

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('{}'.format(name.split('.')[0]))
    ax1.set_title('Percent change')
    ax1.plot(plot_data['change'], label='Actual value')
    ax1.plot(summary['baseline_predict'], label='Baseline (ANN)')
    ax1.plot(summary['pso_predict'], label='PSO+ANN')
    ax1.set_ylabel('Percent change')
    ax1.legend(loc=(1.04, 0.5))

    ax2.set_title('Trading')
    ax2.plot(summary['real_value_change'], label='Actual value')
    ax2.plot(summary['baseline_value_change'], label='Baseline (ANN)')
    ax2.plot(summary['pso_value_change'], label='PSO+ANN')
    ax2.plot(summary['buy_hold'], label='B&H')
    ax2.set_ylabel('Fund')
    ax2.set_xlabel('Number of day')
    ax2.legend(loc=(1.04, 0.5))
    plt.show()
