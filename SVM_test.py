import numpy as np
from sklearn import datasets

from SVM import SVM

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y==0,-1,1)

clf = SVM()
clf.fit(X, y)

print(clf.weights, clf.bias)