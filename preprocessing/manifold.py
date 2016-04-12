import numpy as np


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
