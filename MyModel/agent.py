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
        kw = WindMill.wake_decay
        ct = self.thrust_coeff
        if(x - self.x > 0 and y <= (D + 2*kw)/2):
            return ((1 - np.sqrt(1 - ct))*(D/(D + 2*kw*(x - self.x)))**2)
        return 0

    def __sub__(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        # >= 400m
