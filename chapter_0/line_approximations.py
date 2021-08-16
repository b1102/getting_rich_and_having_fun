import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
sns.set(font_scale=2)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%

activation = 'relu'
batch_size = 128
neurons_number = 64
epochs = 20
optimize = 'Adam'

# %%

# build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(neurons_number, input_shape=(1,), name='input', activation=activation))
model.add(keras.layers.Dense(neurons_number, activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(neurons_number, activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(neurons_number, activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, name='output'))
# summary of the model
model.summary()
model.compile(optimizer=optimize, loss='mean_squared_error', metrics=['mse'])

# %%

X_train = np.linspace(0, 0.8, num=10000)
Y_train = X_train

# %%

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

# %%

X_test = np.linspace(0, 0.8, num=333)
Y_test = X_test

loss, mse = model.evaluate(X_test, Y_test)

# %%

grid = np.linspace(0, 1.0, 150)
predicted = model.predict(grid).reshape((150,))

references = pd.DataFrame({'X': grid, 'Y_reference': grid})
predictions = pd.DataFrame({'X': grid, 'Y_predicted': predicted})

# %%

dims = (30, 20)
fig, ax = pyplot.subplots(figsize=dims)
predicted = model.predict(grid).reshape((150,))

sns.lineplot(x=grid, y=grid, color="red")
sns.lineplot(data=predictions, x=grid, y=predicted, color="blue")
sns.lineplot(data=predictions, x=np.ones(150) * 0.8, y=np.linspace(-0.1, 1.1, 150), color="black")
plt.axvline(0.8, color='k', linestyle='--')
ax.set_title("Loss: {}".format(loss))
fig.legend(labels=['reference', 'predicted', 'cutoff'])
plt.show()
