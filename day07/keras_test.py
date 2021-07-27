#!/usr/bin/env python3

##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 26 Jul 2021                                 #
# Version:	0.0.1                                        #
# What: ? 						                         #
##########################################################
import os
from sysconfig import get_python_version
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
# from torch.autograd import Variable
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader
# from torch import optim

# import tensorflow_datasets as tfds
import tensorflow as tf

# from tensorflow import keras


import keras

import keras.layers as layers

# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# from keras.callbacks import TensorBoard
from keras.datasets import mnist

from sklearn.model_selection import train_test_split

print("Pandas Version: {}".format(pd.__version__))
print("Numpy Version: {}".format(np.__version__))
print("Tensorflow Version: {}".format(tf.__version__))

np.set_printoptions(suppress=True, linewidth=130)


plt.rcParams["figure.figsize"] = (10, 8)


# 2. Split the data into train / validation / test subsets. Make mini-batches, if necesssary.


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Enter data as mnist data set
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# normalizing
x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)


# Test if the shapes are correct and the values make sense
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))
print("x_val shape: " + str(x_val.shape))
print("y_val shape: " + str(y_val.shape))
print("x_test shape: " + str(x_test.shape))
print("y_test shape: " + str(y_test.shape))


# 3. Build the LeNet model


# The LeNet model is build in Keras

model = keras.Sequential()

model.add(
    layers.Conv2D(
        filters=6, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
    )
)
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation="relu"))

model.add(layers.Dense(units=84, activation="relu"))

model.add(layers.Dense(units=10, activation="softmax"))

model.summary()

model.compile(
    loss=keras.metrics.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)


# Fitting the model
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(x_val, y_val),
)
score = model.evaluate(x_test, y_test)
print("Val Loss:", score[0])
print("Val accuracy:", score[1])
