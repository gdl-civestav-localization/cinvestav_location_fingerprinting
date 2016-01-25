import json
import os
__author__ = 'Usuario'

separator = '\t'


def read_file(file_name, mac_list=None):
    if not mac_list:
        mac_list = []
    with open(file_name) as data_file:
        data = json.load(data_file)

    fingerprint = {}
    macs = []
    for wlan in data["features"]:
        mac = wlan["properties"]["MAC"]
        rssi = wlan["properties"]["RSSI"]

        fingerprint[mac] = rssi
        macs.append(mac)
    return fingerprint, list(set(macs + mac_list))


def make_dataset_from_json():
    mac_list = []
    fingerprint_list = []
    result = []
    json_path = 'JSON\\'

    # Read all fingerprint in root directory
    for path, dirs, files in os.walk(json_path):
        # Get current location
        location = path.split("\\")[1].split(",")
        for f in files:
            # Get from file macs and fingerprint
            fingerprint, mac_list = read_file(os.path.join(path, f), mac_list)

            # Append current location and fingerprint
            fingerprint_list.append(fingerprint)
            result.append(location)

    # Generate dataset string
    dataset = []
    for f in fingerprint_list:  # Location and fingerprint
        rssi = []
        for mac in mac_list:
            if mac in f:
                rssi.append(f[mac])  # RSSI of corresponding mac
            else:
                rssi.append(0)  # There is not available rssi for corresponding mac
        dataset.append(rssi)

    write_database(dataset, result, "dataset_house.txt")
    return result, dataset


def write_database(dataset=None, result=None, file_name="database.txt"):
    if not result:
        result = []
    if not dataset:
        dataset = []
    if len(dataset) != len(result):
        raise Exception("Lenght must be equal.")

    # Generate dataset string
    string_builder = []
    for i in range(len(dataset)):  # Location and fingerprint
        string_builder.append("{}{}".format(result[i][0], separator))  # X position
        string_builder.append("{}{}".format(result[i][1], separator))  # Y position
        for rssi in dataset[i]:
            string_builder.append("{}{}".format(rssi, separator))  # Write RSSI
        string_builder.append("\n")

    # Write text file dataset
    with open("dataset\\" + file_name, "w") as text_file:
        text_file.write("".join(string_builder))


def read_dataset():
    # dataset_name = "dataset\\dataset_simulation_zero_iterations.txt"
    dataset_name = "dataset/dataset_simulation2.txt"
    # dataset_name = "dataset\\dataset_house.txt"
    # dataset_name = "dataset\\dataset.txt"

    dataset = []
    result = []
    with open(dataset_name, "r") as text_file:
        lines = text_file.readlines()
        for s in lines:
            values = s.split('\t')
            d = [float(values[0]), float(values[1])]
            rssi = []
            for i in range(2, len(values) - 1, 1):
                val = float(values[i])
                if val == 0:
                    rssi.append(val)  # np.nan)
                else:
                    rssi.append(val)
            dataset.append(rssi)
            result.append(d)
    return result, dataset

if __name__ == '__main__':
    make_dataset_from_json()