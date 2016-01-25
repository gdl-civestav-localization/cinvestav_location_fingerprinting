from sklearn.cluster import KMeans as Km
import numpy as np
import random
import copy


class KMean:
    def __init__(self):
        pass

    def run(self, dataset, n_clusters, max_iters=30, threshold=1.0):
        # Calculate centroids
        _centers = [dataset[random.randint(0, len(dataset))] for i in xrange(n_clusters)]
        _clusters = []

        iteration = 0
        while iteration < max_iters:
            _clusters = [np.NaN] * n_clusters
            for x in dataset:

                # Calculate the cluster of x
                index = 0
                min_dist = self.distance(x, _centers[0])
                for i in xrange(len(_centers)):
                    curr_dist = self.distance(x, _centers[i])
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        index = i
                if _clusters[index] is np.NaN:
                    _clusters[index] = [x]
                else:
                    _clusters[index].append(x)

            centers_old = copy.deepcopy(_centers)

            # Rebuilt centroids
            for i in xrange(len(_centers)):
                _centers[i] = sum(_clusters[i]) / len(_clusters[i])

            # Stop criteria
            if self.stop(centers_old, _centers, threshold):
                break
            iteration += 1
        _centers = np.array(_centers)
        return _clusters, _centers

    def stop(self, centers_old, _centers, threshold=1.0):
        stop = True
        d = 0.0
        for i in xrange(len(_centers)):
            for j in xrange(len(_centers[0])):
                if centers_old[i][j] != _centers[i][j]:
                    stop = False
                    break
            d += self.distance(centers_old[i], _centers[i])
        if d < threshold:
            stop = True
        return stop

    def distance(self, x1, x2):
        d = 0.0
        for i in xrange(len(x1)):
            d += (x1[i] - x2[i])**2
        return d ** 0.5


def get_clouster_centers(dataset, k=2):
    return Km(n_clusters=k).fit(dataset).cluster_centers_