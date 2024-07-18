from NinaPro_Utility import *

import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger
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


db1_path = 'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1\\S1_A1_E3.csv'
data  = get_data_DB1_single(db1_path)
train_reps = [1,3,4,6,8,9]
test_reps = [2,5,7,10]
data = normalise(data, train_reps)
data = filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
data = rectify(data)
gestures = [i for i in range(1,24)]
win_len = 12
win_stride = 1
#data[:1000].plot(figsize = (15,10))
#plt.show()
X_train, y_train, r_train = windowingIMG(data, train_reps, gestures, win_len, win_stride)
print("X_train")
print( X_train.shape)
X_test, y_test, r_test = windowingIMG(data, test_reps, gestures, win_len, win_stride)
print("X_test")
print(X_test.shape)
#y_train = get_categorical(y_train)
#y_test = get_categorical(y_test)

input_shape = (win_len, 10, 1)
model = Sequential()
model.add(Input(input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(23, activation='softmax'))

# Print the model summary
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


# Adjust the batch size
batch_size = 32

# Fit the model with fewer epochs and larger batch size

history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.6)#callbacks=[early_stopping]
results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss values
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#with E3 data test loss, test acc: [2.728247880935669, 0.16291595995426178]
# with new layers test loss, test acc: [2.8902149200439453, 0.15815623104572296]

