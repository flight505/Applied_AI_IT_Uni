#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 13 Jul 2021                                 #
# Version:	0.0.1                                        #
# What: ?						                         #
##########################################################

import os
import numpy as np
import matplotlib.pyplot as plt

os.system("cls||clear")  # this line clears the screen 'cls' = windows 'clear' = unix
# import sys - and sys.exit() to break out
from pprint import pprint  # - use pprint() to pretty print


def plot_1():
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.sin(x ** 2)

    plt.plot(x, y)
    plt.show()


plot_1()
