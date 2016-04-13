def get_LLE(data, num_components=2, n_neighbors=8):
    """
    Perform a LLE transformation
    Parameters
    ----------
    data: Values to transform
    num_components: Number of dimension of the data
    n_neighbors: Number of neighbors
    """
    from sklearn import manifold
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=num_components, method='standard')

    x_lle = clf.fit_transform(data)
    return x_lle


def get_TSNE(data, num_components=2):
    """
    Perform a T-SNE transformation
    Parameters
    ----------
    data: Values to transform
    num_components: Number of dimension of the data
    """
    from sklearn import manifold
    clf = manifold.TSNE(n_components=num_components, init='pca', random_state=0)

    x_lle = clf.fit_transform(data)
    return x_lle
