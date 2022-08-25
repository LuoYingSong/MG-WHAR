import os
import zipfile
import argparse
import numpy as np
import shutil

from io import BytesIO
from pandas import Series
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import signal

from util.util import paint, plot_pie, sliding_window

CONFIG = {
        'opportunity_5imu': {'zip_name': 'OpportunityUCIDataset.zip',
                          'train_runs': ['S1-Drill', 'S1-ADL1', 'S1-ADL3', 'S1-ADL4', 'S1-ADL5', 'S2-Drill', 'S2-ADL1',
                                         'S2-ADL2', 'S2-ADL3', 'S3-Drill', 'S3-ADL1', 'S3-ADL2', 'S3-ADL3', 'S4-Drill',
                                         'S4-ADL1', 'S4-ADL2', 'S4-ADL3', 'S4-ADL4', 'S4-ADL5'],
                          'test_runs': ['S1-ADL2'],
                          'val_runs': ['S2-ADL4', 'S2-ADL5', 'S3-ADL4', 'S3-ADL5'],
                          'train_files': 'OpportunityUCIDataset/dataset/{}.dat',
                          'test_files': 'OpportunityUCIDataset/dataset/{}.dat',
                          'val_files': 'OpportunityUCIDataset/dataset/{}.dat',
                          'invalid': [[1,37], [46, 50], [59, 63], [72, 76], [85, 89], [98, 249]],
                          'x_range': [i for i in range(1, 9 * 5 + 1)],
                          'y': -1,
                          'index_to_label': [[0, -1], [406516, 0], [406517, 1], [404516, 2], [404517, 3], [406520, 4], [404520, 5],
                                             [406505, 6], [404505, 7], [406519, 8], [
                                                 404519, 9], [406511, 10], [404511, 11],
                                             [406508, 12], [404508, 13], [408512, 14], [407521, 15], [405506, 16]],
                          'train_test_split_type': 1
                          }
          }

def select_subject(dataset_name):
    # Test set for the opportunity challenge.
    # below is CNN AND LSTM paper settings
    train_runs = CONFIG[dataset_name]['train_runs']
    test_runs = CONFIG[dataset_name]['test_runs']
    val_runs = CONFIG[dataset_name]['val_runs']
    train_files = [CONFIG[dataset_name]
                   ['train_files'].format(run) for run in train_runs]
    val_files = [CONFIG[dataset_name]
                 ['val_files'].format(run) for run in val_runs]
    test_files = [CONFIG[dataset_name]
                  ['test_files'].format(run) for run in test_runs]
    return train_files, test_files, val_files


def select_columns_opp(data, dataset_name):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """
    invalids = []
    for invalid in CONFIG[dataset_name]['invalid']:
        invalids.append(np.arange(invalid[0], invalid[1]))
    invalid = np.concatenate(invalids)
    return np.delete(data, invalid, 1)


def normalize(data, mean, std):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param mean: numpy integer array
        Array containing mean values for each sensor channel
    :param std: numpy integer array
        Array containing the standard deviation of each sensor channel
    :return:
        Normalized sensor data
    """
    return (data - mean) / std


def divide_x_y(data, dataset_name):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    x_range = CONFIG[dataset_name]['x_range']
    # print(data.shape)
    data_x = data[:, x_range]
    data_y = data[:, CONFIG[dataset_name]['y']]
    return data_x, data_y


def adjust_idx_labels(data_y, dataset_name):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """
    for k, v in CONFIG[dataset_name]['index_to_label']:
        data_y[data_y == k] = v
    return data_y


def process_dataset_file(dataset_name, data):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """
    # Select correct columns
    data = select_columns_opp(data, dataset_name)

    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, dataset_name)
    data_y = adjust_idx_labels(data_y, dataset_name)
    data_y = data_y.astype(int)
    invalid_win = np.nonzero(data_y < 0)[0]  # 小于0的将会被干掉
    data_x = np.delete(data_x, invalid_win, axis=0)
    data_y = np.delete(data_y, invalid_win, axis=0)
    # Replace trailing NaN values with 0.0
    data_x = pd.DataFrame(data_x)
    for column in data_x:
        ind = data_x[column].last_valid_index()
        data_x[column][ind:] = data_x[column][ind:].fillna(0.0)
    data_x = data_x.to_numpy()

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    return data_x, data_y


def generate_data(dataset, args):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """
    # label = args.label
    channel_numbers = len(CONFIG[args.dataset_name]['x_range'])
    train_files, test_files, val_files = select_subject(args.dataset_name)
    print(dataset)
    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')

    try:
        os.makedirs('data/dataset/{}'.format(args.save_path))

    except FileExistsError:  # Remove data if already there.
        for file in os.scandir('data/dataset/{}'.format(args.save_path)):
            if 'data' in file.name:
                os.remove(file.path)

    data_x = np.empty((0, channel_numbers))
    data_y = np.empty(0, dtype=np.uint8)
    if CONFIG[args.dataset_name]['train_test_split_type'] == 1:
        # Generate training files
        print('Generating training files')
        for i, filename in enumerate(train_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> train_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        mean_train = np.mean(data_x, axis=0)
        std_train = np.std(data_x, axis=0)

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/train_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate validation files
        print('Generating validation files')
        for i, filename in enumerate(val_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> val_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/val_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate testing files
        print('Generating testing files')
        for i, filename in enumerate(test_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> test_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/test_data.npz', data=data_x, target=data_y)
    elif CONFIG[args.dataset_name]['train_test_split_type'] == 2:
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate training files
        print('Generating training files')
        for i, filename in enumerate(train_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> train_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))
        mean_train = np.mean(data_x, axis=0)
        std_train = np.std(data_x, axis=0)

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/train_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate validation files
        print('Generating validation files')
        for i, filename in enumerate(val_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> val_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/val_data.npz', data=data_x, target=data_y)
        # Generate testing files
        print('Generating testing files')
    else:
        raise NotImplementedError(
            f"train_test_split_type = {CONFIG[args.dataset_name]['train_test_split_type']} not implemented")


def generate_data_realworld(dataset, args):
    """Function to read the Realworld raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param args
        The input args.
    """
    channel_numbers = len(CONFIG[args.dataset_name]['x_range'])
    train_files, test_files, val_files = select_subject(args.dataset_name)
    print(dataset)
    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')

    try:
        os.makedirs('data/dataset/{}'.format(args.save_path))

    except FileExistsError:  # Remove data if already there.
        for file in os.scandir('data/dataset/{}'.format(args.save_path)):
            if 'data' in file.name:
                os.remove(file.path)

    data_x = np.empty((0, channel_numbers))
    data_y = np.empty(0, dtype=np.uint8)
    if CONFIG[args.dataset_name]['train_test_split_type'] == 1:
        # Generate training files
        print('Generating training files')
        for i, filename in enumerate(train_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> train_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        mean_train = np.mean(data_x, axis=0)
        std_train = np.std(data_x, axis=0)

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/train_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate validation files
        print('Generating validation files')
        for i, filename in enumerate(val_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> val_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/val_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate testing files
        print('Generating testing files')
        for i, filename in enumerate(test_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> test_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data, )
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/test_data.npz', data=data_x, target=data_y)
    elif CONFIG[args.dataset_name]['train_test_split_type'] == 2:
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate training files
        print('Generating training files')
        for i, filename in enumerate(train_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> train_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))
        mean_train = np.mean(data_x, axis=0)
        std_train = np.std(data_x, axis=0)

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/train_data.npz', data=data_x, target=data_y)
        data_x = np.empty((0, channel_numbers))
        data_y = np.empty(0, dtype=np.uint8)
        # Generate validation files
        print('Generating validation files')
        for i, filename in enumerate(val_files):
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {} -> val_data'.format(filename))
                x, y = process_dataset_file(args.dataset_name, data)
                data_x = np.vstack((data_x, x))
                data_y = np.concatenate([data_y, y])
            except KeyError:
                print('ERROR: Did not find {} in zip file'.format(filename))

        data_x = normalize(data_x, mean_train, std_train)

        np.savez_compressed(
            f'data/dataset/{args.save_path}/val_data.npz', data=data_x, target=data_y)
        # Generate testing files
        print('Generating testing files')
    else:
        raise NotImplementedError(
            f"train_test_split_type = {CONFIG[args.dataset_name]['train_test_split_type']} not implemented")

def resample(x, length):
    tmp_x = []
    x_t = np.transpose(x)
    for i in range(x_t.shape[0]):
        tmp_x.append(signal.resample(x_t[i], length))

    return np.transpose(np.array(tmp_x))

def read_all_file_to_pandas(zf):
    df_list = []
    for file_name in sorted(zf.namelist()):
        if file_name.endwith('csv'):
            df = pd.read_csv(BytesIO(zf.read(file_name)))
            df_list.append(df)
        elif file_name.endwith('zip'):
            df_list += read_all_file_to_pandas(zipfile.ZipFile(BytesIO(zf.read(file_name))))
    return df_list

    
    



def partition(path, window, stride, dataset_name):
    # read raw datasets (sample-level)
    print(f"[*] Reading raw files from {path}")
    dataset_train = np.load(os.path.join(path, "train_data.npz"))
    x_train, y_train = dataset_train["data"], dataset_train["target"]
    dataset_val = np.load(os.path.join(path, "val_data.npz"))
    x_val, y_val = dataset_val["data"], dataset_val["target"]
    # apply sliding window over raw samples and generate segments
    data_train, target_train = sliding_window(x_train, y_train, window, stride)
    data_val, target_val = sliding_window(x_val, y_val, window, stride)
    if CONFIG[dataset_name]['train_test_split_type'] == 1:
        dataset_test = np.load(os.path.join(path, "test_data.npz"))
        x_test, y_test = dataset_test["data"], dataset_test["target"]
        data_test, target_test = sliding_window(x_test, y_test, window, stride)
    elif CONFIG[dataset_name]['train_test_split_type'] == 2:
        data_train, data_test, target_train, target_test = train_test_split(
            data_train, target_train, random_state=2022, test_size=0.2)
    else:
        raise NotImplemented('train_test_split = {} is not implemented'.format(
            CONFIG[dataset_name]['train_test_split_type']))
    # save segments to disk

    # show processed datasets info (segment-level)
    print(
        "[-] Train data : {} {}, target {} {}".format(
            data_train.shape, data_train.dtype, target_train.shape, target_train.dtype
        )
    )
    print(
        "[-] Valid data : {} {}, target {} {}".format(
            data_val.shape, data_val.dtype, target_val.shape, target_val.dtype
        )
    )
    print(
        "[-] Test data : {} {}, target {} {}".format(
            data_test.shape, data_test.dtype, target_test.shape, target_test.dtype
        )
    )
    # save processed datasets (segment-level)
    np.savez_compressed(
        os.path.join(path, "train_data.npz"), data=data_train, target=target_train
    )
    np.savez_compressed(
        os.path.join(path, "val_data.npz"), data=data_val, target=target_val
    )
    np.savez_compressed(
        os.path.join(path, "test_data.npz"), data=data_test, target=target_test
    )
    print("[+] Processed segment datasets successfully saved!")
    print(paint("--" * 50, "blue"))


def find_data(name):
    dataset_dir = 'data/raw/'
    dataset_names = CONFIG[name]['zip_name']
    dataset = dataset_dir + dataset_names

    return dataset


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    parser.add_argument(
        '-d', '--dataset_name', type=str, help='dataset name', default='opportunity')
    parser.add_argument(
        '-s', '--subject', type=str, help='Subject to leave out for testing', required=False, default='test')
    parser.add_argument(
        '-w', '--window_size', type=int, help='Size of sliding window (in samples). Default = 24',
        default=24, required=False)
    parser.add_argument(
        '-ws', '--window_step', type=int, help='Stride of sliding window. Default = 12',
        default=12, required=False)
    parser.add_argument(
        '-D', '--save_path', type=str, help='Directory to save preprocessed data',
        default='opportunity', required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variablesin
    return args


if __name__ == '__main__':
    args = get_args()
    dataset = find_data(args.dataset_name)
    generate_data(dataset, args)
    partition('data/dataset/{}/'.format(args.save_path),
              args.window_size, args.window_step, args.dataset_name)
