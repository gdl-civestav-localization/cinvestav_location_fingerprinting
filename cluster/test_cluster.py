from cluster.Clustering import KMean
from datasets import DatasetManager


def test_kmean():
    train_set, valid_set, test_set = DatasetManager.read_dataset(dataset_name="dataset_simulation_20.csv", shared=False)
    kmean = KMean()

    _clusters, _centers = kmean.run(
        dataset=train_set[0],
        n_clusters=5,
        max_iters=100,
        threshold=1.0
    )

    assert _clusters
    assert _centers is not None


if __name__ == '__main__':
    test_kmean()
