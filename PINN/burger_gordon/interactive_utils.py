"""
Wrappers with widgets to improve the presentation. 
Author: Gordon Erlebacher
Date: 2023-02-01
"""

import ipywidgets as widgets
from ipywidgets import interact
import Fall_2022_Burgers_Equation_functions as impl
from DictX import *
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------
def draw_plot_locations():
    @interact(n=(20,141,20), sz=6, reset=False)
    def make_plot(n, sz, reset):
        dct = GlobDct()
        dct1 = DictX(dct.copy())
        # dct1 = DictX(GlobDct().copy())  # for experimentation
        dct1.n_x = n
        dct1.n_t = n
        dct1.x_np = np.linspace(dct.x_domain[0], dct.x_domain[1], dct1.n_x, dtype=np.float32)
        dct1.t_np = np.linspace(dct.t_domain[0], dct.t_domain[1], dct1.n_t, dtype=np.float32)
        fig, ax = plt.subplots(1,1, figsize=(sz, sz))
        ax = impl.plot_locations(ax, x=dct1.x_np, t=dct1.t_np, **dct1, title=f"Training points {dct1.n_x*dct1.n_t} --> {dct1.n_x*dct1.n_t*3}")
    plt.show()
#-----------------------------------------------------------
def draw_plot_intermediate_values():
    dct = GlobDct()
    dct1 = DictX(dct.copy())
    impl.plot_intermediate_values1(**dct)
    plt.show()
#-----------------------------------------------------------
def draw_plot_extrapolated_solution():
    dct = GlobDct()
    dct1 = DictX(dct.copy())
    impl.plot_extrapolated_solution(**dct)
#-----------------------------------------------------------
