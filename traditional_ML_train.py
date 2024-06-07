import argparse
from datetime import datetime

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pickle

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from dataset_creating import read_data, create_dataset


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

    # Set the seed for NumPy
    np.random.seed(args.random_state)

    print(f'Training the model on the dataset: {args.dataset}')

    # Load the dataset
    df = read_data(args.dataset)  # Load the dataset

    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.num_classes, test_size=args.test_size,
                                                      random_state=args.random_state)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(f'The shape of the training set: {x_train.shape}')
    print(f'The shape of the testing set: {x_test.shape}')
    print(f'The shape of the training labels: {y_train.shape}')
    print(f'The shape of the testing labels: {y_test.shape}')

    # Define classifiers
    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(n_neighbors=6),
        "Linear SVM": SVC(kernel="linear"),
        "RBF SVM": SVC(kernel="rbf"),
        "Gaussian Process": GaussianProcessClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Neural Net": MLPClassifier(),
        "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    # Define parameter grids
    param_grids = {
        "Nearest Neighbors": {'n_neighbors': [3, 5, 7, args.num_classes]},
        "Linear SVM": {'C': [0.1, 1, 10]},
        "RBF SVM": {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'auto']},
        "Gaussian Process": {},  # No hyperparameters to tune for Gaussian Process
        "Decision Tree": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
        "Neural Net": {'hidden_layer_sizes': [(50,), (100,), (50, 50)]},
        "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 1, 10]},
        "Naive Bayes": {},  # No hyperparameters to tune for Naive Bayes
        "QDA": {},  # No hyperparameters to tune for QDA
        'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
        'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    }

    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)
    # Train the models
    results = {}
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for name, classifier in classifiers.items():
        print(f"Grid search for {name}...")
        clf = GridSearchCV(classifier, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
        clf.fit(x_train, y_train)
        results[name] = clf
        print(f"Best parameters for {name}: {clf.best_params_}")

        with open(f"models/{name}_{current_time}.pkl", 'wb') as f:  # saving the model
            pickle.dump(clf, f)

    print("\nEvaluation on test set:")
    for name, clf in results.items():
        y_pred = clf.predict(x_test)
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred))

    print("\nSelected Model:")
    best_model_name = max(results, key=lambda k: results[k].best_score_)
    best_model = results[best_model_name]
    print(f"Model: {best_model_name}")
    print(f"Best parameters: {best_model.best_params_}")
    print(f"Best mean accuracy: {best_model.best_score_}")
    # Print confusion matrix for the best model
    y_pred = best_model.predict(x_test)
    print(f"Confusion matrix for the best model:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
