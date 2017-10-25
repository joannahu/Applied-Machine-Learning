import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten

# Task3: Convolutional Neural Network

#load data
train_SVHN = sio.loadmat("/rigel/edu/coms4995/datasets/train_32x32.mat")
test_SVHN = sio.loadmat("/rigel/edu/coms4995/datasets/test_32x32.mat")
#train_SVHN = sio.loadmat('train_32x32.mat')
#test_SVHN = sio.loadmat('test_32x32.mat')

X_train_SVHN = np.asarray(train_SVHN['X'])
X_test_SVHN = np.asarray(test_SVHN['X'])
y_train_SVHN = np.asarray(train_SVHN['y'])
y_test_SVHN = np.asarray(test_SVHN['y'])

num_classes = 11
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
X_train_images = np.rollaxis(X_train_SVHN, 3, 0)
X_test_images = np.rollaxis(X_test_SVHN, 3, 0)
y_train_binary = to_categorical(y_train_SVHN)
y_test_binary = to_categorical(y_test_SVHN)
X_train_images /= 255.0
X_test_images /= 255.0

# Base Model
from keras.layers import Conv2D, MaxPooling2D, Flatten

num_classes = 11
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile("sgd", "categorical_crossentropy", metrics=['accuracy'])
history_cnn = cnn.fit(X_train_images, y_train_binary,
                      batch_size=128, epochs=50, verbose=1, validation_split=.2)

score_basic = cnn.evaluate(X_test_images, y_test_binary)
print("Task3 base model Test loss: {:.2f}".format(score_basic[0]))
print("Task3 base model Test Accuracy: {:.2f}".format(score_basic[1]))

# Batch Normalization
from keras.layers import BatchNormalization
num_classes = 11
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(8, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(8, (3, 3)))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(64, activation='relu'))
cnn_small_bn.add(Dense(num_classes, activation='softmax'))

cnn_small_bn.compile("sgd", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_small_bn = cnn_small_bn.fit(X_train_images, y_train_binary,
                                        batch_size=128, epochs=50, verbose=1, validation_split=.2)

score_bn = cnn_small_bn.evaluate(X_test_images, y_test_binary)
print("Task3 Batch Normalization Test loss: {:.2f}".format(score_bn[0]))
print("Task3 Batch Normalization Test Accuracy: {:.2f}".format(score_bn[1]))

