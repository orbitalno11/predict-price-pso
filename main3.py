import pandas as pd
import numpy as np
from simulate import Simulator

data = pd.read_csv('data/test_set/FB-18.csv')

simulator = Simulator(10, data, initial_fund=100000)
simulator.start()

summary = simulator.summary()
print(summary)
