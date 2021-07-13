#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 13 Jul 2021                                 #
# Version:	0.0.1                                        #
# What: ? 						                         #
##########################################################

import os

os.system("cls||clear")  # this line clears the screen 'cls' = windows 'clear' = unix
import sys  # - and sys.exit() to break out
import pprint  # pprint # - use pprint() to pretty print


import pandas as pd
import numpy as np
import seaborn as sns

hc = pd.read_csv("day02/hydrocarbon.csv", names=["n", "x", "y"])

pprint(hc.head())


sys.exit()
sns.scatterplot(hc["x"], hc["y"])


def linear_f(x, beta_0, beta_1):
    return x * beta_1 + beta_0
