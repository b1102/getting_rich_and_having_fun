import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
import seaborn as sns
from numpy import sin


def train_data(start=None, end=None, number=None):
    x_test = np.linspace(start, end, num=number)
    y_test = x_test
    return x_test, y_test


def train_data(start=None, end=None, number=None, f=None):
    x_test = np.linspace(start, end, num=number)
    y_test = f(x_test)
    return x_test, y_test


def plot_results(grid, predicted, name, mean_absolute_error, cutoff):
    sns.set_style("darkgrid")
    sns.set(font_scale=2)

    dims = (30, 20)
    fig, ax = pyplot.subplots(figsize=dims)

    sns.lineplot(x=grid, y=grid, color="red")
    sns.lineplot(x=grid, y=predicted, color="blue")
    sns.lineplot(x=np.ones(150) * cutoff, y=np.linspace(-0.1, 1.1, 150), color="black")
    plt.axvline(cutoff, color='k', linestyle='--')
    ax.set_title("{}. Mean absolute error: {}".format(name, mean_absolute_error))
    fig.legend(labels=['reference', 'predicted', 'cutoff'])
    plt.show()


def plot_results(X_train, Y_train, X_predicted, Y_predicted, name, mean_absolute_error, cutoff, f):
    sns.set_style("darkgrid")
    sns.set(font_scale=2)

    dims = (30, 20)
    fig, ax = pyplot.subplots(figsize=dims)

    sns.lineplot(x=X_train, y=Y_train, color="red")
    sns.lineplot(x=X_predicted, y=Y_predicted, color="blue")
    sns.lineplot(x=np.ones(150) * cutoff, y=np.linspace(-0.1, 1.1, 150), color="black")
    none_seen_grid = np.linspace(cutoff, 1.0, 25)
    sns.lineplot(x=none_seen_grid, y=f(none_seen_grid), color="black")
    plt.axvline(cutoff, color='k', linestyle='--')
    ax.set_title("{}. Mean absolute error: {}".format(name, mean_absolute_error))
    fig.legend(labels=['reference', 'predicted', 'part that we aiming to approximate'])
    plt.show()
