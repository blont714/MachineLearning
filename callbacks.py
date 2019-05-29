# -*- coding: utf-8-*-
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from sklearn import model_selection

data_num = 2000
np.random.seed(14)
min_length = 2
max_length = 12
min_num = 0
max_num = 3


def make_data():
    i = 0
    xdata = []
    ydata = []
    while i < data_num:
        length = np.random.randint(min_length, max_length + 1)
        data = np.random.randint(min_num, max_num + 1, length)
        xdata.append(list(data.reshape(length, 1)))
        ydata.append(sum(data))
        i += 1

    return xdata, ydata

xdata, ydata = make_data()

maxlen = max([len(x) for x in xdata])

xdata = sequence.pad_sequences(xdata, maxlen=maxlen)
ydata = np.array(ydata).reshape(data_num, 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    xdata, ydata, test_size=0.1, random_state=0)

in_out_dims = 1
hidden_dims = 16

model = Sequential()
model.add(InputLayer(batch_input_shape=(None, maxlen, in_out_dims)))
model.add(
    SimpleRNN(units=hidden_dims, return_sequences=False))
model.add(Dense(in_out_dims))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

callbacks = [EarlyStopping(patience=0, verbose=1), CSVLogger("simple_RNN_history.csv")]
# fitting
history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_split=0.1, callbacks=callbacks)
