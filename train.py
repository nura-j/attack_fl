import argparse
from sklearn.model_selection import train_test_split
from dataset_creating import read_data, filter_by_column_count


def main():
    parser = argparse.ArgumentParser(description='Training ML models on the dataset for the classification task.')
    parser.add_argument('--dataset', default='data/moore_clean_cols_all.csv', type=str,
                        help='Path to the dataset file.')
    parser.add_argument('--model_name', type=str, help='Name of the model to be saved.')
    parser.add_argument('--test_size', default=0.3, type=float, help='Size of the test set.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state for splitting the dataset.')

    args = parser.parse_args()
    print(f'Training the model on the dataset: {args.dataset}')
    print(f'The model will be saved as: {args.model_name}')

    # Load the dataset
    df = read_data(args.dataset)

    # Filter the dataset by the column count
    df = filter_by_column_count(df, top_n=6)

    # Check if the dataset has null values
    print(f'Number of null values in the dataset: {df.isnull().sum().sum()}')

    # check if the dataset has non-numeric values
    non_numeric = df.select_dtypes(exclude='number')
    print(f'Non-numeric columns: {non_numeric.columns}')

    print(f'The shape of the dataset: {df.shape}\n')
    features = df.drop('266', axis=1)
    target = df['266']
    print(f'The shape of the features: {features.shape}')
    print(f'The shape of the target: {target.shape}')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=args.test_size,
                                                        random_state=args.random_state)
    print(f'The shape of the training features: {X_train.shape}')
    print(f'The shape of the testing features: {X_test.shape}')
    print(f'The shape of the training target: {y_train.shape}')
    print(f'The shape of the testing target: {y_test.shape}')


if __name__ == '__main__':
    main()
