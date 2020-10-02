import numpy as np
import matplotlib.pyplot as plt


class WindMill(object):
    wake_decay = 0.05

    def __init__(self, x, y, thrust_coeff, dia=100):
        self.x = x
        self.y = y
        self.dia = dia
        self.thrust_coeff = thrust_coeff

    def ret_wake(self, x, y):
        D = self.dia
        decay = wake_decay

    def __sub__(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        # >= 400m
