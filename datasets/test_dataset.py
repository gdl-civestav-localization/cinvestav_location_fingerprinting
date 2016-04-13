from datasets.DatasetManager import read_dataset
import Simulation

__author__ = 'Gibran Felix'


def test_dataset_manager():
    assert read_dataset(dataset_name="dataset_simulation_20.csv", shared=False)
    assert read_dataset(dataset_name="dataset_simulation_20.csv", shared=True)


def test_simulation_generate_dataset():
    Simulation.plt.ion()
    Simulation.run_generate_dataset()


def test_simulation_generate_coverage_img():
    Simulation.plt.ion()
    Simulation.run_generate_img()


if __name__ == "__main__":
    test_simulation_generate_coverage_img()
    test_simulation_generate_dataset()
    test_dataset_manager()
