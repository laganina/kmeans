import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
#.data is all of the features, brings all of them as 0 and 1

y = digits.target

#k = len(np.unique(y)), centroids
k = 10

#the amount of numbers that we have that we're gonna classify

samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, estimator.inertia_,
            metrics.homogeneity_score(y, estimator.labels_),
            metrics.completeness_score(y, estimator.labels_),
            metrics.v_measure_score(y, estimator.labels_),
            metrics.adjusted_rand_score(y, estimator.labels_),
            metrics.adjusted_mutual_info_score(y, estimator.labels_),
            metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))

#train a ton of clf using this funct and uses a bunch of dif things to score
#we compare y values to the labels that estimator gives us

clf = KMeans(n_clusters=k,init='random',n_init=10)
#init changes location of different centroids
#n_init, how many times we run the algorithm with different centroid seeds
#10 is default value

bench_k_means(clf,'1',data)
