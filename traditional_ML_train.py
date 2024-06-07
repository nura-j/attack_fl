import argparse
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from dataset_creating import read_data, filter_by_column_count, create_dataset


def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.diag(cm)


def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    print(f'Per class accuracy: {per_class_accuracy(y_test, y_pred)}')
    print(f'Confusion matrix: {confusion_matrix(y_test, y_pred)}')

    return model


def train_models(models, x_train, y_train, x_test, y_test):
    for model in models:
        print(f'Training the model: {model}')
        if model == 'SVM':
            model = train_model(SVC(), x_train, y_train, x_test, y_test)
        elif model == 'RandomForest':
            model = train_model(RandomForestClassifier(), x_train, y_train, x_test, y_test)
        elif model == 'KNN':
            model = train_model(KNeighborsClassifier(), x_train, y_train, x_test, y_test)
        else:
            print(f'Model: {model} is not supported.')
    return models

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
    parser.add_argument('--models', default='[SVM, RandomForest, KNN]', type=str,
                        help='List of models to train on the dataset.')

    args = parser.parse_args()
    print(f'Training the model on the dataset: {args.dataset}')

    # Load the dataset
    df = read_data(args.dataset) # Load the dataset

    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.num_classes, test_size=args.test_size, random_state=args.random_state)
    print(f'The shape of the training set: {x_train.shape}')
    print(f'The shape of the testing set: {x_test.shape}')
    print(f'The shape of the training labels: {y_train.shape}')
    print(f'The shape of the testing labels: {y_test.shape}')

    # Train the models




if __name__ == '__main__':
    main()
