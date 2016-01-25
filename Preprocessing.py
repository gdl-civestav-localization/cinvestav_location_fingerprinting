import DatasetManager
from sklearn.preprocessing import Imputer
from sklearn.decomposition import RandomizedPCA
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Usuario'


# Perform a softmax scaling
def get_softmax_normalization(data=None, t=1.0):
    if not data:
        data = []

    e = np.exp(np.array(data) / t)
    dist = e / np.sum(e)
    return dist


# Perform a logistic scaling
def get_logistic_normalization(data=None):
    if not data:
        data = []

    data = 1 / (1 + np.exp(-1 * np.array(data)))
    return data


# Perform a gaussian scaling
def get_gaussian_normalization(data=np.array([])):
    data = (data - np.mean(data, 0)) / np.std(data, 0)
    return data


# Perform a linear scaling
def get_linear_normalization(data=np.array([])):
    min_value = np.min(data, 0)
    max_value = np.max(data, 0)
    data = (data - min_value) / (max_value - min_value)
    return data


# Solve missing values
def solve_missing_values(data=None):
    if not data:
        data = []

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit_transform(data)
    return data


# Perform a LLE transformation
def get_LLE(data=np.array([]), num_components=2, n_neighbors=8):
    from sklearn import manifold
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=num_components, method='standard')

    x_lle = clf.fit_transform(data)
    return x_lle


# Perform a T-SNE transformation
def get_TSNE(data=np.array([]), num_components=2):
    from sklearn import manifold
    clf = manifold.TSNE(n_components=num_components, init='pca', random_state=0)

    x_lle = clf.fit_transform(data)
    return x_lle


# Perform a PCA transformation
def get_pca(data=np.array([]), num_components=2):
    pca = RandomizedPCA(n_components=num_components, whiten=False)
    data = pca.fit_transform(data)
    return data, pca.explained_variance_ratio_


def run_experiment():
    result, dataset = DatasetManager.read_dataset()

    # Reduce to a 2D dimensionality for plotting the data
    dataset = get_gaussian_normalization(dataset)
    # dataset = get_LLE(dataset, num_components=2, n_neighbors=80)
    dataset, explained_variance_ratio_ = get_pca(dataset, num_components=2)

    plot_embedding(dataset, result)


def plot_embedding(dataset, result, title=None):
    dataset = get_linear_normalization(dataset)
    used_labels = []
    ax = plt.subplot(111)
    shown_images = [[1., 1.]]
    for c in xrange(len(result)):
        if result[c] not in used_labels:
            used_labels.append(result[c])
            cluster = []
            for i in xrange(len(dataset)):
                if result[i] == result[c]:
                    cluster.append(dataset[i])
            plt.plot(*zip(*cluster), marker='o')

            dist = np.sum((cluster[0] - shown_images) ** 2, 1)
            if np.min(dist) < 1e-3:
                continue
            from matplotlib.offsetbox import TextArea, AnnotationBbox
            text = '(' + str(int(result[c][0])) + ',' + str(int(result[c][1])) + ')'
            text_area = TextArea(text, minimumdescent=False)
            ab = AnnotationBbox(text_area, cluster[0],
                                xybox=(-20, 40),
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->"))
            ax.add_artist(ab)
            shown_images.append(cluster[0])
    plt.show()


if __name__ == '__main__':
    run_experiment()

    # result, dataset = DatasetManager.read_dataset()
    # dataset = get_gaussian_normalization(dataset)
    # print dataset
