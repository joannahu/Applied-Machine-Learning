#Task 4

import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras import applications
from keras.applications.vgg16 import preprocess_input
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split,GridSearchCV
import tensorflow as tfvation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import keras

#load data
imageFolder = "/rigel/edu/coms4995/datasets/pets"
imagesNames = [image.load_img(os.path.join(imageFolder, filename), target_size=(224, 224))
               for filename in os.listdir(imageFolder) if filename.endswith(".jpg")]
X = np.array([image.img_to_array(img) for img in imagesNames])

model = applications.VGG16(include_top=False, weights='imagenet')

X_pre = preprocess_input(X)
features = model.predict(X_pre)

num = features.shape[0]

features_ = features.reshape(num, -1)

y = np.zeros(num, dtype='int')
y_num = int(num/2)
y[y_num:] = 1
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)

# Logistic Regression
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV().fit(X_train, y_train)
print("LogisticRegression train score: {:.3f}".format(lr.score(X_train, y_train)))
print("LogisticRegression test score: {:.3f}".format(lr.score(X_test, y_test)))

# SGDClassifier

from sklearn.linear_model import SGDClassifier
sgd_c = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False).fit(X_train, y_train)

sgd_c = sgd_c.fit(X_train, y_train)

print("SGDClassifier train score: {:.3f}".format(sgd_c.score(X_train, y_train)))
print("SGDClassifier test score: {:.3f}".format(sgd_c.score(X_test, y_test)))



