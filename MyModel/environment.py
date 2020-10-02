import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


from agent import WindMill
from function_approx import Stirling


class Farm(object):

    def __init__(self, n_mills, length, breadth, wind_prop, margin, init_der, power_curve_dir):

        self.n_mills = n_mills
        self.length = length
        self.breadth = breadth
        self.margin = margin
        self.wind_prop = wind_prop
        # wind-prop - dict, keys -> wind speed and wind
        self.coords = pd.read_csv(init_der).values  # n X 2
        self.Wmills = WindMill(self.coords[:, 0], self.coords[:, 1], )
        self.power_curve = pd.read_csv(power_curve_dir).values
        n = self.power_curve.shape[0]
        self.thrust_coefficient = Stirling(
            self.power_curve[:, 0], self.power_curve[:, 1], self.wind_prop['speed'], n)
        self.power = Stirling(self.power_curve[:, 0],
                              self.power_curve[:, 2], self.wind_prop['speed'], n)

        Wmills = Windmills(self.coords[:, 0], self.coords[:, 1], self.thrust_coeff)
        self.Wmills = Wmills

    def net_wake(self, x, y, Wmills):
        wake_dict = dict()
        for wm_i in self.Wmills:
            for wm_j in self.Wmills:
                wake_dict[wm_i].append(wm_i.ret_wake(wm_j))
