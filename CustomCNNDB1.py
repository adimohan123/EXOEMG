from NinaPro_Utility import *

import tensorflow as tf
from keras.regularizers import l2
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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



db1_path = 'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1\\E3Data.csv'
#TODO: detect Rest
data  = get_data(db1_path)
train_reps = [1,3,4,6,7,8,9]
test_reps = [2,5,10]
#data = normalise(data, train_reps) #sus
data = filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
#data = rectify(data)

np.unique(data.stimulus)
gestures = [i for i in range(1,24)]
win_len = 150
win_stride = 30
data[:1000].plot(figsize = (15,10))
plt.show()
X_train, y_train, r_train = windowing(data, train_reps, gestures, win_len, win_stride)
X_test, y_test, r_test = windowing(data, test_reps, gestures, win_len, win_stride)
y_train = get_categorical(y_train)
y_test = get_categorical(y_test)
#nodes = X_train.shape[1]
model = Sequential()
input_shape = (win_len, 10, 1)

# Block 1
# Block 1
model.add(Conv2D(32, kernel_size=(1, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 2
model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(3, 1)))
model.add(Dropout(0.5))

# Block 3
model.add(Conv2D(64, kernel_size=(5, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(3, 1)))
model.add(Dropout(0.5))

# Block 4
model.add(Conv2D(64, kernel_size=(5, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 5
model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Final layer with softmax activation
model.add(Flatten())
model.add(Dense(23, activation='softmax'))



# Print the model summary
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Adjust the batch size
batch_size = 32

# Fit the model with fewer epochs and larger batch size
history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)
#test loss, test acc: [2.8804447650909424, 0.14748743176460266] with dense layers
#test loss, test acc: [5.009033679962158, 0.3956112861633301] with 150 win len  gth and 30 stride
#test loss, test acc: [3.130826473236084, 0.4532458782196045] epoch 100
#test loss, test acc: [3.634037494659424, 0.49848636984825134]

#with E3 data test loss, test acc: [2.728247880935669, 0.16291595995426178]
# with new layers test loss, test acc: [2.8902149200439453, 0.15815623104572296]

