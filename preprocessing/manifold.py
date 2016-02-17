import numpy as np
from matplotlib import pyplot as plt

from datasets import DatasetManager
from preprocessing.dimensionality_reduction import get_pca
from preprocessing.scaling import get_gaussian_normalization, get_linear_normalization


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


def run_experiment():
    dataset, result = DatasetManager.read_dataset()

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