import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

from preparation import Preparation
from indicator import Indicator
from ANN import ANN


class C_Model:

    def __init__(self):
        self.read_data = pd.read_csv('../data/test_set')