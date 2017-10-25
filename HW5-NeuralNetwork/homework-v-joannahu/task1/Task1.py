import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn import datasets
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import keras

# Task1: Two hidden layers

#model selection
def make_model(optimizer="sgd", hidden_size=32):
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Dense(32, activation='relu'),
        Dense(3),
        Activation('softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

param_grid = {'epochs': [10, 20, 50],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}

grid = GridSearchCV(clf, param_grid=param_grid)

#load data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=60)

# convert class vectors to binary class matrices
num_classes = 3
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#train model
grid.fit(X_train, y_train)

res = pd.DataFrame(grid.cv_results_)
table = res.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])

print (table)

#evaluate
score = grid.score(X_test, y_test)
print("\nTask1 Test Accuracy: {:.3f}".format(score))
