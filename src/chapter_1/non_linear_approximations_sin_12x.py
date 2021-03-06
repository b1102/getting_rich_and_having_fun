import numpy as np
import tensorflow as tf
from numpy import sin, exp
from tensorflow import keras
import os

from utils.utils import train_data, plot_results



# %%
name = "sin(12 * x)"
activation = 'relu'
batch_size = 64
neurons_number = 128
epochs = 50
optimize = 'Adam'
function_to_approximate = lambda x: sin(12 * x)

# %%

# build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(neurons_number, input_shape=(1,), name='input', activation=activation))
model.add(keras.layers.Dense(neurons_number, activation=activation))
model.add(keras.layers.Dense(neurons_number, activation=activation))
model.add(keras.layers.Dense(neurons_number, activation=activation))
model.add(keras.layers.Dense(neurons_number, activation=activation))
model.add(keras.layers.Dense(1, name='output'))
# summary of the model
model.summary()
model.compile(optimizer=optimize, loss='mean_squared_error', metrics=['mse', 'mean_absolute_error'])

# %%
X_train, Y_train = train_data(start=0, end=0.8, number=100000, f=function_to_approximate)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

# %%
X_test, Y_test = train_data(start=0, end=1.0, number=333, f=function_to_approximate)
loss, mse, mean_absolute_error = model.evaluate(X_test, Y_test)

# %%
X_predicted = np.linspace(0, 1.0, 150)
Y_predicted = model.predict(X_predicted).reshape((150,))

plot_results(X_train, Y_train, X_predicted, Y_predicted, name, mean_absolute_error, 0.8, function_to_approximate)
