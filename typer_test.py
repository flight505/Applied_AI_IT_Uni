#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 12 Aug 2021                                 #
# Version:	0.0.1                                        #
# What: ? 						                         #
##########################################################

import os
import typer

os.system("cls||clear")  # this line clears the screen 'cls' = windows 'clear' = unix
# import sys - and sys.exit() to break out
# from pprint import pprint - use pprint() to pretty print

app = typer.Typer()


@app.command()
def hello(name: str, iq: int, display_iq: bool = True):
    print(f"Hello!! {name}")
    if display_iq:
        print(f"Your IQ is {iq}")


@app.command()
def goodbye():
    print("goodbye!!")


if __name__ == "__main__":
    app()
