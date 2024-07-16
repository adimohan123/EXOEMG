from NinaPro_Utility import *

import tensorflow.compat.v1 as tf

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from functools import reduce
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# np_load_old = np.load

# # modify the default parameters of np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

db2_path = 'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB2\\DB2_s1'

data  = get_data(db2_path,'S1_E1_A1.mat')
train_reps = [1,3,4,6]
test_reps = [2,5]
data = normalise(data, train_reps) #sus
data = filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
data = rectify(data)
data[:1000].plot(figsize = (15,10))
np.unique(data.stimulus)
gestures = [i for i in range(1,18)]
win_len = 600
win_stride = 20
X_train, y_train, r_train = windowing(data, train_reps, gestures, win_len, win_stride)
X_test, y_test, r_test = windowing(data, test_reps, gestures, win_len, win_stride)
y_train = get_categorical(y_train)
y_test = get_categorical(y_test)


def get_model(X_train):
    nodes = X_train.shape[1]

    inputs = Input(shape=(nodes, 12))
    LSTM_1 = LSTM(nodes, dropout=0.2, return_sequences=True)(inputs)
    LSTM_2 = LSTM(nodes, dropout=0.2, return_sequences=True)(LSTM_1)
    LSTM_3 = LSTM(nodes, dropout=0.2, return_sequences=True)(LSTM_2)
    LSTM_4 = LSTM(nodes, dropout=0.2, return_sequences=True)(LSTM_3)

    x = Flatten()(LSTM_4)

    dense1 = Dense(128, activation='sigmoid')(x)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='sigmoid')(dense1)
    dropout2 = Dropout(0.5)(dense2)
    predictions = Dense(17, activation='softmax')(dropout2)

    model = Model(inputs=inputs, outputs=predictions)

    return model

model = get_model(X_train)
histories, model = train_model(model, X_train, y_train, X_test, y_test, save_to= 'temp', epoch = 20)
