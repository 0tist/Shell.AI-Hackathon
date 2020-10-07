import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from numba import njit
from fastai.vision.all import *
from math import radians as DegToRad       # Degrees to radians Conversion
from tqdm import tqdm                          # For Progressbar

from shapely.geometry import Point             # Used in constraint checking
from shapely.geometry.polygon import Polygon

from agent import WindMill
# from function_approx import Stirling


class Farm(object):

    def __init__(self, length, breadth, diam,
                 margin, power_curve_dir, turb_loc_dir, wind_data_year):

        self.length = length  # length of the farm
        self.breadth = breadth  # breadth of the farm
        self.margin = margin  # margin of the farm
        # self.wind_prop = wind_prop # prop of the wind
        # wind-prop - dict, keys -> wind speed and wind
        self.power_curve = pd.read_csv(power_curve_dir).values
        self.turb_loc = pd.read_csv(turb_loc_dir).values  # n X 2
        self.wind_data_year = wind_data_year
        self.n_mills = self.turb_loc.shape[0]
        # self.Wmills = WindMill(self.turb_loc[:, 0], self.turb_loc[:, 1], )
        self.diam = diam  # diameter of the turbine

    # def net_wake(self, x, y, Wmills):
    #     wake_dict = dict()
    #     for wm_i in self.Wmills:
    #         for wm_j in self.Wmills:
    #             wake_dict[wm_i].append(wm_i.ret_wake(wm_j))
    #
    #     self.wake_dict = wake_dict
    #     return self.wake_dict
    #
    # def combined_wake(self):
    #     net_wake_dict = dict()
    #     for m in self.wake_dict:
    #         net_wake_dict[m] = np.sqrt(sum(np.square(np.array(self.wake_dict[m].values))))
    #
    #     self.net_wake_dict = net_wake_dict
    #     return self.net_wake_dict
    #
    # def net_wind_speed(self):
    #     windEff_dict = dict()
    #     i = 0
    #     for m in self.net_wake_dict:
    #         windEff_dict[m] = self.power[i, 0]*(1 - self.net_wake_dict[m])
    #         i += 1
    #
    #     self.windEff_dict = windEff_dict
    #     return self.windEff_dict

    def binWindResourceData(self):

        df = pd.read_csv(self.wind_data_year)
        wind_resource = df[['drct', 'sped']].to_numpy(dtype=np.float32)

        slices_drct = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
        # slices_drct   = [360, 10.0, 20.0.......340, 350]
        n_slices_drct = slices_drct.shape[0]

        # speed 'slices'
        slices_sped = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                       18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        n_slices_sped = len(slices_sped)-1

        # placeholder for binned wind
        binned_wind = np.zeros((n_slices_drct, n_slices_sped),
                               dtype=np.float32)

        # 'trap' data points inside the bins.
        for i in range(n_slices_drct):
            for j in range(n_slices_sped):

                # because we already have drct in the multiples of 10
                foo = wind_resource[(wind_resource[:, 0] == slices_drct[i])]

                foo = foo[(foo[:, 1] >= slices_sped[j])
                          & (foo[:, 1] < slices_sped[j+1])]

                binned_wind[i, j] = foo.shape[0]

        self.wind_inst_freq = binned_wind/np.sum(binned_wind)
        # print(self.wind_inst_freq)
        return self.wind_inst_freq

    @staticmethod
    def rotatedFrame(turb_coords, wind_drct):

        wind_drct = wind_drct - 90

        wind_drct = DegToRad(wind_drct)
        cos_dir = np.cos(wind_drct)
        sin_dir = np.sin(wind_drct)

        rotate_coords = np.zeros((turb_coords.shape[0], 2), dtype=np.float32)
        rotate_coords[:, 0] = (turb_coords[:, 0] * cos_dir) - (turb_coords[:, 1] * sin_dir)
        rotate_coords[:, 1] = (turb_coords[:, 0] * sin_dir) + (turb_coords[:, 1] * cos_dir)

        rotate_coords = rotate_coords

        return (rotate_coords)

    def jensenParkWave(self, wind_sped, wind_drct):

        # WIND_SPED

        turb_radius = self.diam / 2
        idx_foo = np.argmin(np.abs(self.power_curve[:, 0] - wind_sped))

        C_t = self.power_curve[idx_foo, 1]

        kw = 0.05

        impact_on_ibyj = np.zeros((self.n_mills, self.n_mills), dtype=np.float32)

        self.rotate_coords = Farm.rotatedFrame(self.turb_loc, wind_drct)

        for i in range(self.n_mills):
            for j in range(self.n_mills):
                x = self.rotate_coords[i, 0] - self.rotate_coords[j, 0]
                y = self.rotate_coords[i, 1] - self.rotate_coords[j, 1]

                if i != j:
                    if x <= 0 or np.abs(y) > (turb_radius + kw*x):
                        impact_on_ibyj[i, j] = 0.0

                    else:
                        impact_on_ibyj[i, j] = (1 - np.sqrt(1 - C_t)) * \
                            ((turb_radius / (turb_radius + kw*x))**2)

                # print(impact_on_ibyj[i, j])

        self.wake_matrix = impact_on_ibyj

        sped_deficit = np.sqrt(np.sum(impact_on_ibyj**2, axis=1))
        self.sped_deficit = sped_deficit

        return self.sped_deficit

    def partAEP(self, wind_sped, wind_drct):

        rotate_coords = Farm.rotatedFrame(self.turb_loc, wind_drct)
        sped_dfct = self.jensenParkWave(wind_sped, wind_drct)

        turb_pwr = np.zeros(self.n_mills, dtype=np.float32)

        for i in range(self.n_mills):
            wind_sped_eff = wind_sped*(1.0 - self.sped_deficit[i])

            idx_foo = np.argmin(np.abs(self.power_curve[:, 0] - wind_sped_eff))
            pwr = self.power_curve[idx_foo, 2]

            turb_pwr[i] = pwr

        self.power = np.sum(turb_pwr)

        return self.power

    def totalAEP(self):

        n_turbs = self.turb_loc.shape[0]
        assert n_turbs == 50, "Error! Number of turbines is not 50."

        slices_drct = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
        n_slices_drct = slices_drct.shape[0]

        slices_sped = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                       18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]

        n_slices_sped = len(slices_sped)-1

        # print(self.wind_inst_freq.shape)
        # print("*****************************")

        farm_pwr = np.zeros((self.wind_inst_freq.shape), dtype=np.float32)

        for i in tqdm(range(n_slices_drct), disable=False):
            for j in range(n_slices_sped):

                wind_drct = slices_drct[i]
                wind_sped = (slices_sped[j] + slices_sped[j+1]) / 2

                pwr = self.partAEP(wind_drct, wind_sped)
                farm_pwr[i, j] = pwr

        farm_pwr = self.wind_inst_freq * farm_pwr

        farm_pwr = np.sum(farm_pwr)

        year_hours = 365.*24.
        AEP = year_hours * farm_pwr

        self.AEP = AEP/1e3

        return self.AEP

    def checkConstraints(self):
        bound_clrnc = 50
        prox_constr_viol = False
        peri_constr_viol = False

        # create a shapely polygon object of the wind farm
        farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
        farm_poly = Polygon(farm_peri)

        # checks if for every turbine perimeter constraint is satisfied.
        # breaks out if False anywhere
        for turb in self.turb_loc:
            turb = Point(turb)
            inside_farm = farm_poly.contains(turb)
            correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
            if (inside_farm == False or correct_clrnc == False):
                peri_constr_viol = True
                break

        # checks if for every turbines proximity constraint is satisfied.
        # breaks out if False anywhere
        for i, turb1 in enumerate(self.turb_loc):
            for turb2 in np.delete(self.turb_loc, i, axis=0):
                if np.linalg.norm(turb1 - turb2) < 4*self.diam:
                    prox_constr_viol = True
                    break

        # print messages
        if peri_constr_viol == True and prox_constr_viol == True:
            print('Somewhere both perimeter constraint and proximity constraint are violated\n')
        elif peri_constr_viol == True and prox_constr_viol == False:
            print('Somewhere perimeter constraint is violated\n')
        elif peri_constr_viol == False and prox_constr_viol == True:
            print('Somewhere proximity constraint is violated\n')
        else:
            print('Both perimeter and proximity constraints are satisfied !!\n')

        return()
