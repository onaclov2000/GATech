from __future__ import print_function
from loader import Loader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ld = Loader()

plt_X = []
plt_Y = []

[X, y] = ld.load_data('datasets/Curious_George_train_features_100_percent.csv')

for n_clusters in range(1, 16):
    km = KMeans(n_clusters=n_clusters)
    km.fit(X)
    plt_X.append(n_clusters)
    plt_Y.append(km.inertia_)

plt.plot(plt_X, plt_Y)
plt.ylabel('Within groups sum of squares')
plt.xlabel('Number of Clusters')
plt.savefig('figures/' + 'Kmeans_Curious_George_Elbow_Curve.png')

