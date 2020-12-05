import pandas as pd
import numpy as np
from simulate import Simulator

data = pd.read_csv('data/test_set/FB-18.csv')

simulator = Simulator(simulate_day=10, simulate_data=data, baseline_model=None, pso_model=None)
simulator.start()

summary = simulator.summary()
print(summary)
