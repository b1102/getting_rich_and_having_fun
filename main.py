import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
from matplotlib import pyplot

activation = 'relu'
batch_size = 128
neurons_number = 128
epochs = 20
optimize = 'Adam'

# build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(neurons_number, input_shape=(1,), name='input', activation=activation))
model.add(keras.layers.Dense(neurons_number, name='dense_layer_2', activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(neurons_number, name='dense_layer_3', activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(neurons_number, name='dense_layer_4', activation=activation))
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, name='output'))
# summary of the model
model.summary()
model.compile(optimizer=optimize, loss='mean_squared_error', metrics=['mse'])

X_train = np.linspace(0, 0.8, num=10000)
Y_train = X_train

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

X_test = np.linspace(0, 0.8, num=333)
Y_test = X_test

test_loss, test_acc = model.evaluate(X_test, Y_test)

to_predict = np.linspace(0, 1.0, 150)
predict = model.predict(to_predict).reshape((150,))
predictions = pd.DataFrame({'X_test': to_predict, 'Y_test': predict}, columns=['X_test', 'Y_test'])

training = pd.DataFrame({'X_test': X_train, 'Y_test': Y_train}, columns=['X_test', 'Y_test'])

dims = (30, 20)
fig, ax = pyplot.subplots(figsize=dims)

sns.lineplot(data=predictions, x="X_test", y="Y_test", color="red")
sns.lineplot(data=training, x="X_test", y="Y_test", color="blue")
plt.show()
