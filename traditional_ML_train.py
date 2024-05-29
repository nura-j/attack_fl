import argparse
from sklearn.model_selection import train_test_split
from dataset_creating import read_data, filter_by_column_count, create_dataset


def main():
    parser = argparse.ArgumentParser(description='Training ML models on the dataset for the classification task.')
    parser.add_argument('--dataset', default='data/moore_clean_cols_all.csv', type=str,
                        help='Path to the dataset file.')
    parser.add_argument('--test_size', default=0.2, type=float,
                        help='Size of the test set.')
    parser.add_argument('--num_classes', default=6, type=int,
                        help='Number of classes to keep in the dataset.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state for splitting the dataset.')

    args = parser.parse_args()
    print(f'Training the model on the dataset: {args.dataset}')

    # Load the dataset
    df = read_data(args.dataset) # Load the dataset

    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.num_classes, test_size=args.test_size, random_state=args.random_state)
    print(f'The shape of the training set: {x_train.shape}')
    print(f'The shape of the testing set: {x_test.shape}')
    print(f'The shape of the training labels: {y_train.shape}')
    print(f'The shape of the testing labels: {y_test.shape}')

    # Train the model



if __name__ == '__main__':
    main()
