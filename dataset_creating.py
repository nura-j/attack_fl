import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical
pd.set_option('display.max_columns', None)


def read_data(path):
    return pd.read_csv(path)


def application_to_numeric(x):
    if x == 'WWW':
        return 0  # WWW
    elif x == 'MAIL':
        return 1  # MAIL
    elif x == 'FTP-DATA':
        return 2  # BULK
    elif x == 'FTP-CONTROL':
        return 2  # BULK
    elif x == 'FTP-PASV':
        return 2  # BULK
    elif x == 'DATABASE':
        return 3  # DB
    elif x == 'SERVICES':
        return 4  # SERV
    elif x == 'P2P':
        return 5  # P2P
    elif x == 'ATTACK':
        return 6  # ATTACK
    elif x == 'INTERACTIVE':
        return 7  # INTERACTIVE
    elif x == 'MULTIMEDIA':
        return 8  # MULTIMEDIA
    elif x == 'GAMES':
        return 9  # GAMES
    else:
        return -1


def numeric_to_application(x):
    if x == 0:
        return 'WWW'
    elif x == 1:
        return 'MAIL'
    elif x == 2:
        return 'BULK'
    elif x == 3:
        return 'DB'
    elif x == 4:
        return 'SERV'
    elif x == 5:
        return 'P2P'
    elif x == 6:
        return 'ATTACK'
    elif x == 7:
        return 'INTERACTIVE'
    elif x == 8:
        return 'MULTIMEDIA'
    elif x == 9:
        return 'GAMES'
    else:
        return 'UNKNOWN'


def replace_values_in_column(df, old_values, new_values, column_name=None):
    #  Check if the old_values and new_values have the same length
    if len(old_values) != len(new_values):
        raise ValueError("old_values and new_values must have the same length")

    #  Check if the column_name is None
    if column_name is None:
        # go through all the columns
        for col in df.columns:
            df[col] = df[col].replace(old_values, new_values)
    else:
        # Replace old values with new values in the identified columns
        for col in column_name:
            for old_val, new_val in zip(old_values, new_values):
                df[col] = df[col].str.replace(old_val, new_val)
    return df


def filter_by_column_count(df, classes_count=None, top_n=6, by_threshold=False):
    # Get the classes count
    if classes_count is None:
        classes_count = dict(df['class'].value_counts()) # or 266

    if by_threshold:
        average = sum(classes_count.values()) / len(classes_count)
        threshold = average / 10
        columns_to_remove = []
        for key, value in classes_count.items():
            if value < threshold:
                columns_to_remove.append(key)
        print(f'Columns to remove: {columns_to_remove}')
        return df[~df['class'].isin(columns_to_remove)] # or 266
    else:
        # Sort the dictionary by values
        sorted_classes_count = dict(sorted(classes_count.items(), key=lambda item: item[1], reverse=True))
        # Get the top n classes
        top_n_classes = dict(list(sorted_classes_count.items())[:top_n])
        print(f'Top {top_n} classes: {top_n_classes}')
        # Filter the dataset by the column count
        columns_to_remove = []
        for key, value in classes_count.items():
            if key not in top_n_classes.keys():
                columns_to_remove.append(key)
        print(f'Columns to remove: {columns_to_remove}')
        return df[~df['class'].isin(columns_to_remove)] # or 266


def create_dataset(df, top_n=6, test_size=0.2, random_state=42):
    # Filter the dataset by the column count
    df = filter_by_column_count(df, top_n=top_n)
    features = df.drop('class', axis=1) # or 266
    target = df['class']
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                        random_state=random_state)
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = to_categorical(y_train, num_classes=top_n)  # Convert to one-hot encoding
    y_test = to_categorical(y_test, num_classes=top_n)  # Convert to one-hot encoding

    # Reshaping the NumPy array to meet TensorFlow standards
    x_train = np.reshape(x_train, (x_train.shape[0],
                                            x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0],
                                            x_test.shape[1], 1))
    return x_train, x_test, y_train, y_test

def oversample(df):
    over_sample = RandomOverSampler(sampling_strategy='all', random_state=42)
    x = df.drop('class', axis=1) #or # or 266
    y = df['class'] # or # or 266
    x_over, y_over = over_sample.fit_resample(x, y)
    return pd.concat([x_over, y_over], axis=1)


def undersample(df):
    under_sample = RandomUnderSampler(sampling_strategy='all', random_state=42)
    x = df.drop('class', axis=1) # or 266
    y = df['class'] # or 266
    x_under, y_under = under_sample.fit_resample(x, y)
    return pd.concat([x_under, y_under], axis=1)

def generate_iid_client_data(X, y, num_clients=10): #  IID data distribution
    """
    Generate IID client data by splitting X and y into equal parts for each client.

    Parameters:
    - X (array-like): Features.
    - y (array-like): Labels.
    - num_clients (int): Number of clients.

    Returns:
    - client_X (list): List of feature subsets for each client.
    - client_y (list): List of label subsets for each client.
    """

    def split_data(X, y, num_clients):
        client_size = len(X) // num_clients
        X, y = shuffle(X, y)
        client_X = [X[i:i + client_size] for i in range(0, client_size * num_clients, client_size)]
        client_y = [y[i:i + client_size] for i in range(0, client_size * num_clients, client_size)]
        return client_X, client_y

    return split_data(X, y, num_clients)
# change the main name to reflect the name of the file and their purpose


def generate_noniid_client_shards(X_train, y_train, num_shards, shard_size, num_clients=10,
                                  min_shard=1, max_shard=30):
                                    # the function of creating data into non-IID and unequal shreds
    """
    Sample non-I.I.D client data from network dataset s.t. clients
    have unequal amount of data|
    :param X_train: training data
    :param y_train: training labels
    :param num_shards: number of shards to divide the data into
    :param shard_size: size of each shard
    :param num_clients:
    :param min_shard: minimum number of shards assigned to a client
    :param max_shard: maximum number of shards assigned to a client
    :returns set of client x data, client y data - training & test
    """

    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * shard_size)
    labels = np.argmax(y_train, axis=1)  # Convert the one-hot encoding to labels


    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide the shards into random chunks for every client s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards).astype(int)

    # Ensure sum of random_shard_size does not exceed num_shards
    if sum(random_shard_size) > num_shards:
        random_shard_size -= 1

    # Assign the shards to each client
    for i in range(num_clients):
        if len(idx_shard) == 0:
            break
        shard_count = min(random_shard_size[i], len(idx_shard))
        rand_set = set(np.random.choice(idx_shard, shard_count, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * shard_size:(rand + 1) * shard_size]), axis=0)

    # Distribute any remaining shards
    while len(idx_shard) > 0:
        min_client = min(dict_users, key=lambda x: len(dict_users[x]))
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[min_client] = np.concatenate(
                (dict_users[min_client], idxs[rand * shard_size:(rand + 1) * shard_size]), axis=0)

    client_X = [X_train[value.astype(int)] for key, value in dict_users.items()]
    client_y = [y_train[value.astype(int)] for key, value in dict_users.items()]

    return client_X, client_y

def calculate_shards_and_rows_simple(total_samples, desired_shard_size):
    num_shards = total_samples // desired_shard_size
    num_rows = desired_shard_size
    return num_shards, num_rows


def calculate_shards_and_rows(total_samples, desired_num_shards=None, min_shard_size=50, max_shard_size=500):
    """
    Calculate the number of shards and rows per shard based on total samples and constraints.

    Parameters:
    - total_samples (int): The total number of samples in the dataset.
    - desired_num_shards (int, optional): The desired number of shards. If None, will be calculated dynamically.
    - min_shard_size (int): The minimum size of a shard.
    - max_shard_size (int): The maximum size of a shard.

    Returns:
    - num_shards (int): The number of shards.
    - shard_size (int): The number of rows per shard.
    """

    if desired_num_shards:
        shard_size = max(min_shard_size, min(max_shard_size, total_samples // desired_num_shards))
    else:
        # Determine a shard size dynamically within the given range
        possible_shard_sizes = [size for size in range(min_shard_size, max_shard_size + 1) if total_samples % size == 0]
        if possible_shard_sizes:
            shard_size = max(possible_shard_sizes)
        else:
            shard_size = max(min_shard_size, min(max_shard_size, total_samples // (total_samples // max_shard_size)))

    num_shards = total_samples // shard_size
    return num_shards, shard_size

def main():
    parser = argparse.ArgumentParser(description='Reading and filtering the dataset.')
    parser.add_argument('--path', default='data/all_data.csv', type=str, help='Path to the dataset.')
    parser.add_argument('--top_n', default=6, type=int, help='Number of classes to keep in the dataset.')
    parser.add_argument('--distribution', default='iid', type=str, choices=['iid', 'noniid'],)
    parser.add_argument('--num_clients', default=10, type=int, help='Number of clients for non-IID distribution.')
    args = parser.parse_args()

    path = args.path
    df = read_data(path)
    print(f'The shape of the dataset: {df.shape}\n')
    # Filtering the dataset - without dropping nan - simple cleaning and converting the text classes to numeric
    # 1. Converting the text classes to numeric
    classes_count = dict(df['266'].value_counts()) # or 266
    print(f'The classes count (text): {classes_count}')
    df['266'] = df['266'].apply(application_to_numeric)  # Applying the function to the column
    classes_count = dict(df['266'].value_counts())
    print(f'The classes count (numeric): {classes_count}\n')

    # 2. Checking the data types of the columns
    data_type_series = df.dtypes
    cols = []
    # print('Data type of each column of Dataframe:')
    for col, type_ in data_type_series.items():
        if type_ == 'object':
            cols.append(col)
    #         print(f'{col} : {type_}', end=', ')
    # print('\n')
    df = replace_values_in_column(df, ['Y', 'N'], ['1', '0'], column_name=cols)
    df = replace_values_in_column(df, ['?'], [np.nan])

    # 3.1 remove the missing values in the rows
    df_rows = df.dropna()

    # checking the distribution of the classes
    classes_count_rows = dict(df_rows['266'].value_counts()) # or class

    # 3.2 remove the missing values in the columns
    df_cols = df.dropna(axis='columns')

    # checking the distribution of the classes
    classes_count_cols = dict(df_cols['266'].value_counts()) # or class

    # Save the datasets
    # 1. Save the dataset with the rows
    df_rows.to_csv('data/moore_clean_rows_all.csv', index=False)

    # 2. Save the dataset with the columns
    df_cols.to_csv('data/moore_clean_cols_all.csv', index=False)

    # 3. Save the dataset with the rows and columns
    df.to_csv('data/cleaned_data.csv', index=False)

    # Printing each dataframe information
    print(f'The shape of the dataset with rows: {df_rows.shape}')
    print(f'The classes count (text) with rows: {classes_count_rows}')

    print(f'The shape of the dataset with columns: {df_cols.shape}')
    print(f'The classes count (text) with columns: {classes_count_cols}')

    print(f'The shape of the dataset with rows and columns: {df.shape}')



if __name__ == '__main__':
    main()
