import pandas as pd
import argparse


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
    elif x == 'MULTIMEDIA':
        return 7  # MULTIMEDIA
    elif x == 'INTERACTIVE':
        return 8  # INT
    elif x == 'GAMES':
        return 9  # GAMES
    else:
        return -1


def main():
    pd.set_option('max_columns', None)  # Display all columns
    parser = argparse.ArgumentParser(description='Reading and filtering the dataset.')
    parser.add_argument('--path', default='data/all_data.csv', type=str, help='Path to the dataset.')
    args = parser.parse_args()

    path = args.path
    df = read_data(path)

    classes_count = dict(df['266'].value_counts())
    print(f'The classes count (text): {classes_count}')
    df['266'] = df['266'].apply(application_to_numeric)  # Applying the function to the column
    classes_count = dict(df['266'].value_counts())
    print(f'The classes count (numeric): {classes_count}')
    dataTypeSeries = df.dtypes
    cols = []
    print('Data type of each column of Dataframe :')
    for col, type_ in dataTypeSeries.items():
        if type_ == 'object':
            print(col)
            cols.append(col)


if __name__ == '__main__':
    main()

