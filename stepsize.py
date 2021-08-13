import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import os
from sklearn.model_selection import train_test_split

print(f"the present working directory is: {os.getcwd()}")
DATA_PATH = os.path.join("data")
PROJECT_ROOT_DIR = "."


def load_model_data(data):
    pwd = os.getcwd()
    filepath = os.path.join(pwd, DATA_PATH, data)
    return pd.read_csv(filepath)


data = load_model_data("heart.csv")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cost(x, y, Theta):
    n = x.shape[0]  # number of samples
    h = sigmoid(np.dot(x, Theta))
    return -(1.0 / n) * np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))


# The dependent variable is selected as y
y = data["target"]
# The independent variables are selected. The address is discarted, as it is strings and does not make sense to include in the linear regression
x = data.drop(["target"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

variables = x_train.shape[1]
theta = np.zeros(variables, dtype=float)
init_cost = cost(x_train, y_train, theta)
print(f"The initial cost is: {init_cost:.2f}")


def gradient(x, y, Theta) -> float:
    n = x.shape[0]  # number of samples
    h = sigmoid(np.dot(x, Theta))
    return 1.0 / n * np.dot(x.T, (h - y))


mu = 0.00005  # stepsize
variables = x_train.shape[1]
init_theta = np.zeros(variables, dtype=float)  # initializing theta as zeroes
init_cost = cost(x_train, y_train, init_theta)

new_cost = 0
m = 200000
i = 0

while (new_cost < init_cost) and (i < m):
    init_cost = cost
    i += 1
    gd_theta = gradient(x_train, y_train, init_theta)
    updated_weights = init_theta - gd_theta * mu
    new_cost = cost(x_train, y_train, updated_weights)

    if i % (m / 20) == 0:
        print(new_cost)
    init_theta = updated_weights
