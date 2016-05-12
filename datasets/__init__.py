import os
import pandas as pd

__author__ = 'Gibran Felix'


def format_samples(log_folder='anyplace'):
    macs = set()
    samples = []
    all_rss = {}
    x = 0
    y = 0
    floor = 0

    for file_name in os.listdir(log_folder):
        with open(os.path.join(log_folder, file_name), 'rb') as f:

            # Read document
            f.readline()  # First line is junk
            for line in f.readlines():
                # Check if is label
                if line[0] != '#':
                    values = line.split(' ')

                    # Get labels
                    x = values[1]
                    y = values[2]
                    floor = values[6]

                    # Get mac and rss values
                    mac = values[4]
                    rss = values[5]
                    macs.add(mac)
                    all_rss[mac] = rss

                else:  # If is label add row
                    samples.append({
                        'x': x,
                        'y': y,
                        'z': floor,
                        'rss': all_rss
                    })
                    all_rss = {}
            # Add the last row
            samples.append({
                'x': x,
                'y': y,
                'z': floor,
                'rss': all_rss
            })
    return samples, macs


def parse_anyplace_log_to_dataset(dataset_name='cinvestav', separator='\t', with_z_label=False):
    # Format anyplace log files
    samples, macs = format_samples()

    # Generate dataset
    dataset = []
    for sample in samples:
        row = []
        for mac in macs:
            if mac in sample['rss']:
                row.append(sample['rss'][mac])
            else:
                row.append(0)

        # Add labels
        row.append(sample['x'])
        row.append(sample['y'])
        if with_z_label:
            row.append(sample['z'])

        # Add row to dataset
        dataset.append(row)

    # Generate columns name
    columns = []
    for ap in range(0, len(macs)):
        columns.append("ap{}".format(ap))
    columns.append("result_x")
    columns.append("result_y")
    if with_z_label:
        columns.append("result_z")

    if with_z_label:
        dataset_name += '_z'

    dataset_name = os.path.join(os.path.dirname(__file__), 'dataset', dataset_name + '.csv')

    df = pd.DataFrame(data=dataset, columns=columns)
    df.to_csv(dataset_name, sep=separator, encoding='utf-8')


if __name__ == "__main__":
    # from datasets.DatasetManager import read_dataset
    # read_dataset()

    parse_anyplace_log_to_dataset(
        dataset_name='cinvestav_testbed',
        separator=',',
        with_z_label=True
    )
