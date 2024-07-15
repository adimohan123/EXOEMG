from NinaPro_Utility import *

import tensorflow as tf

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
plt.show()
X_train, y_train, r_train = windowing(data, train_reps, gestures, win_len, win_stride)
X_test, y_test, r_test = windowing(data, test_reps, gestures, win_len, win_stride)
y_train = get_categorical(y_train)
y_test = get_categorical(y_test)
print('check point 1 ')
nodes = X_train.shape[1]
print('Check point 2')
model = Sequential()

# Add LSTM layer with dropout
model.add(LSTM(nodes, return_sequences=False))
model.add(Dropout(0.2))

# Flatten the output
model.add(Flatten())

# Add Dense layers with ReLU activation
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add output layer with softmax activation
model.add(Dense(17, activation='softmax'))  # Assuming 10 classes for output, adjust as necessary

# Print the model summary
model.summary()


