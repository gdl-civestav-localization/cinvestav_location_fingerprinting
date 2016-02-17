import numpy as np
from sklearn.decomposition import RandomizedPCA


# Perform a PCA transformation
def get_pca(data=np.array([]), num_components=2):
    pca = RandomizedPCA(n_components=num_components, whiten=False)
    data = pca.fit_transform(data)
    return data, pca.explained_variance_ratio_
