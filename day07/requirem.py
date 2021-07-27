#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 26 Jul 2021                                 #
# Version:	0.0.1                                        #
# What: ? 						                         #
##########################################################

import os

os.system("cls||clear")  # this line clears the screen 'cls' = windows 'clear' = unix
# import sys - and sys.exit() to break out
# from pprint import pprint - use pprint() to pretty print


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# with open(
# "/Users/jvang/Documents/Projects/Applied_AI_IT_Uni/requirements copy.txt", "r"
# ) as myfile:
# data = myfile.read().splitlines()
#
# print(data)
#
def remove_text_after_char(infile) -> "str":
    with open(infile) as infile:
        with open(
            "file_out.txt",
            "w",
        ) as outfile:
            for line in infile.readlines():
                return line.strip("==").split("==", -1)[0] + "\n"


remove_text_after_char(
    "/Users/jvang/Documents/Projects/Applied_AI_IT_Uni/requirements copy.txt"
)
