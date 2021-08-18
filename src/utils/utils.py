import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
import seaborn as sns


def train_data(start=None, end=None, number=None):
    x_test = np.linspace(start, end, num=number)
    y_test = x_test
    return x_test, y_test


def plot_results(grid, predicted, name, mean_absolute_error, cutoff):
    sns.set_style("darkgrid")
    sns.set(font_scale=2)

    predictions = pd.DataFrame({'X': grid, 'Y_predicted': predicted})
    # %%
    dims = (30, 20)
    fig, ax = pyplot.subplots(figsize=dims)

    sns.lineplot(x=grid, y=grid, color="red")
    sns.lineplot(data=predictions, x=grid, y=predicted, color="blue")
    sns.lineplot(data=predictions, x=np.ones(150) * cutoff, y=np.linspace(-0.1, 1.1, 150), color="black")
    plt.axvline(cutoff, color='k', linestyle='--')
    ax.set_title("{}. Mean absolute error: {}".format(name, mean_absolute_error))
    fig.legend(labels=['reference', 'predicted', 'cutoff'])
    plt.show()