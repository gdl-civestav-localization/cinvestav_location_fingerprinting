from sklearn.decomposition import RandomizedPCA


def get_pca(data, num_components=2):
    """
    Perform a PCA transformation
    Parameters
    ----------
    data: Values to transform
    num_components: Number of dimension of the data
    """
    pca = RandomizedPCA(n_components=num_components, whiten=False)
    data = pca.fit_transform(data)
    return data, pca.explained_variance_ratio_
