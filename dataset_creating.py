import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
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
        classes_count = dict(df['266'].value_counts())

    if by_threshold:
        average = sum(classes_count.values()) / len(classes_count)
        threshold = average / 10
        columns_to_remove = []
        for key, value in classes_count.items():
            if value < threshold:
                columns_to_remove.append(key)
        print(f'Columns to remove: {columns_to_remove}')
        return df[~df['266'].isin(columns_to_remove)]
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
        return df[~df['266'].isin(columns_to_remove)]


def create_dataset(df, top_n=6, test_size=0.2, random_state=42):
    # Filter the dataset by the column count
    df = filter_by_column_count(df, top_n=top_n)
    features = df.drop('266', axis=1)
    target = df['266']
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test

# change the main name to reflect the name of the file and their purpose
def main():
    parser = argparse.ArgumentParser(description='Reading and filtering the dataset.')
    parser.add_argument('--path', default='data/all_data.csv', type=str, help='Path to the dataset.')
    args = parser.parse_args()

    path = args.path
    df = read_data(path)
    print(f'The shape of the dataset: {df.shape}\n')
    # Filtering the dataset - without dropping nan - simple cleaning and converting the text classes to numeric
    # 1. Converting the text classes to numeric
    classes_count = dict(df['266'].value_counts())
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
    classes_count_rows = dict(df_rows['266'].value_counts())

    # 3.2 remove the missing values in the columns
    df_cols = df.dropna(axis='columns')

    # checking the distribution of the classes
    classes_count_cols = dict(df_cols['266'].value_counts())

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
    print(f'The classes count (text) with rows and columns: {classes_count}')


if __name__ == '__main__':
    main()
