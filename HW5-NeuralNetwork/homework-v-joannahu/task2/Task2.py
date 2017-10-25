import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
import tensorflow as tfdatasets
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import keras


# Task2: model dropout vs vanilla

#load data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#vanilla model selection
def make_model_vanilla(optimizer="sgd", hidden_size=32):
    model = Sequential([
        Dense(32, input_shape=(784,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model_vanilla)

param_grid = {'epochs': [10, 20, 50],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}

grid = GridSearchCV(clf, param_grid=param_grid)

history = grid.fit(X_train, y_train)

#Plot Vanilla Model and calculate score
model = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128,
                    epochs=50, verbose=1, validation_split=.1)

df = pd.DataFrame(history.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

score_vanilla = model.evaluate(X_test, y_test, verbose=0)
print("Task2 Vanilla Test Loss: {:.3f}".format(score_vanilla[0]))
print("Task2 Vanilla Test Accuracy: {:.3f}".format(score_vanilla[1]))




#dropout model selection
from keras.layers import Dropout

def make_model_dropout(optimizer="sgd", hidden_size=32):
    model_dropout = Sequential([
        Dense(32, input_shape=(784,), activation='relu'),
        Dropout(.5),
        Dense(32, activation='relu'),
        Dropout(.5),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model_vanilla)

param_grid = {'epochs': [10, 20, 50],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}

grid = GridSearchCV(clf, param_grid=param_grid)
history_dropout = grid.fit(X_train, y_train)

#Plot Dropout Model and calculte score
from keras.layers import Dropout

model_dropout = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dropout(.5),
    Dense(32, activation='relu'),
    Dropout(.5),
    Dense(10, activation='softmax'),
])
model_dropout.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_dropout = model_dropout.fit(X_train, y_train, batch_size=128,
                            epochs=50, verbose=1, validation_split=.1)

df = pd.DataFrame(history_dropout.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

score_dropout = model_dropout.evaluate(X_test, y_test, verbose=0)
print("Task2 Dropout Test Loss: {:.3f}".format(score_dropout[0]))
print("Task2 Dropout Test Accuracy: {:.3f}".format(score_dropout[1]))