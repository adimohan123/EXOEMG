import scipy.io
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
from keras.layers import Input, Dense, ZeroPadding2D, Dropout, Activation, Flatten, Conv2D, Conv1D, MaxPooling2D, AveragePooling2D, concatenate, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
from NinaPro_Utility import *
import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger
from keras.regularizers import l2
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




'''file = h5py.File("C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1_S1_image.h5",'r')
imageData   = file['imageData'][:]
imageLabel  = file['imageLabel'][:]
file.close()'''

db1_path = 'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1\\S1_A1_E3.csv'
data  = get_data_DB1_single(db1_path)
train_reps = [1,3,4,6,8,9]
test_reps = [2,5,7,10]
ALl_reps = [1,2,3,4,5,6,7,8,9,10]
data = normalise(data, train_reps)
data = filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
data = rectify(data)
gestures = [i for i in range(1,24)]
win_len = 12
win_stride = 12

imageData, imageLabel, r_train = windowingIMG(data, ALl_reps, gestures, win_len, win_stride)



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# prepare data
n = imageData.shape[0]
idx = np.random.permutation(n)
data  = imageData[idx]
'''
label = imageLabel[idx]

data  = np.expand_dims(data, axis=3)
label = convert_to_one_hot(label, 23).T

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = 0.2, random_state = 42)

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'epoch':[]}
        self.accuracy = {'epoch':[]}
        self.val_loss = {'epoch':[]}
        self.val_acc = {'epoch':[]}

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


eps = 200
'''

'''
def CNN(input_shape, classes):
    X_input = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X_input)
    X = Activation('relu', name='relu1')(X)

    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = Activation('relu', name='relu2')(X)
    X = AveragePooling2D((3, 3), strides=(2, 2), name='pool1')(X)

    X = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv3')(X)
    X = Activation('relu', name='relu3')(X)
    X = AveragePooling2D((3, 3), strides=(2, 2), name='pool2')(X)

    X = Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding='same', name='conv4')(X)
    X = Activation('relu', name='relu4')(X)

    X = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv5')(X)

    X = ZeroPadding2D((0, 1))(X)

    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='CNN')
    return model

model = CNN(input_shape = (12, 10, 1), classes = 52)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = LossHistory()

model.fit(data, label, validation_split=0.2, epochs=eps, batch_size=256, verbose=1, callbacks=[history])

print('-------------------------------------------------------------------------')
preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))
history.loss_plot('epoch')
'''