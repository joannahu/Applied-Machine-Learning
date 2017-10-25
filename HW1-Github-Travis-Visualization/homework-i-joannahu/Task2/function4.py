import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def fun4(x,y):
	kmean = KNeighborsClassifier()
	return np.mean(cross_val_score(kmean, x, y,cv=5))