def run_experiment():
    from datasets import DatasetManager
    from preprocessing.scaling import get_gaussian_normalization
    from preprocessing.dimensionality_reduction import get_pca

    train_set, valid_set, test_set = DatasetManager.read_dataset()
    dataset, result = train_set

    # Reduce to a 2D dimensionality for plotting the data
    dataset = get_gaussian_normalization(dataset)
    # dataset = get_LLE(dataset, num_components=2, n_neighbors=80)
    dataset, explained_variance_ratio_ = get_pca(dataset, num_components=2)

    plot_embedding(dataset, result)


def plot_embedding(dataset, result):
    """
    Plot clusters with same labels and label as legends
    Parameters
    ----------
    dataset: Values to plot
    result: Labels of the data
    """
    import numpy as np
    from preprocessing.scaling import get_linear_normalization
    from matplotlib import pyplot as plt

    dataset = np.asarray(dataset)
    result = np.asarray(result)

    dataset = get_linear_normalization(dataset)

    used_labels = []
    ax = plt.subplot(111)
    shown_images = [[1., 1.]]
    for c in xrange(len(result)):
        if result[c].tolist() not in used_labels:
            used_labels.append(result[c].tolist())
            cluster = []
            for i in xrange(len(dataset)):
                if np.array_equal(result[i], result[c]):
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
