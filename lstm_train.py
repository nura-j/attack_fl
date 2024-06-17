import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import operator
import os
import glob
import matplotlib.pyplot as plt

import copy
import random
from numpy.random import randint, rand
from random import sample

#Importing our TensorFlow & Keras libraries
import keras
import tensorflow as tf
# from keras.layers.convolutional import Conv1D
from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import to_categorical
# from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from dataset_creating import read_data, create_dataset, oversample


# the function of creating data into non-IID and unequal shreds
def noniid_unequal(X_train, y_train, X_test, y_test, num_clients=10):
    """
    Sample non-I.I.D client data from network dataset s.t. clients
    have unequal amount of data|
    :param dataset:
    :param num_clients:
    :returns set of client x data, client y data - training & test
    """
    # 297347 training sample --> 145 rows/shard X 2307 shards (customized for this dataset)
    num_shards, num_rows = 2307, 145
    idx_shard = [i for i in range(num_shards)]  # list of number of shreds
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_rows)
    labels = np.argmax(y_train, axis=1)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_clients)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_rows:(rand + 1) * num_rows]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_rows:(rand + 1) * num_rows]),
                    axis=0)
    else:

        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_rows:(rand + 1) * num_rows]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_rows:(rand + 1) * num_rows]),
                    axis=0)

    client_X = []  # list of clients features
    client_y = []  # list of clients labels
    for key, value in dict_users.items():
        temp = [int(x) for x in value]
        client_X.append(X_train[temp])
        client_y.append(y_train[temp])

    # repeat the process for the testset or go iid for the test set (here we opt to have iid for testing )
    client_size_test = len(X_test) // num_clients
    X_test, y_test = shuffle(X_test, y_test)
    client_X_test = [X_test[i:i + client_size_test] for i in range(0, client_size_test * num_clients, client_size_test)]
    client_y_test = [y_test[i:i + client_size_test] for i in range(0, client_size_test * num_clients, client_size_test)]

    return client_X, client_y, client_X_test, client_y_test


class RNNModel1:
    def __init__(self, x_training_data, num_classes=6):
        self.model = Sequential()
        self.model.add(LSTM(units=100, input_shape=(x_training_data.shape[1], 1)))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_and_evaluate(self, x_train, x_test, y_train, y_test, batch_size=1000, epochs=10, validation_split=0.1):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        accuracy = self.model.evaluate(x_test, y_test)[1]
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        # per_class_accuracy = np.mean(y_test == np.round(self.model.predict(x_test)), axis=0)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        return accuracy, per_class_accuracy, cm

class RNNModel2:
    def __init__(self, X_train, num_classes=6):
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
        self.model.add(LSTM(units=100))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_and_evaluate(self, x_train, x_test, y_train, y_test, batch_size=1000, epochs=10, validation_split=0.1):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        accuracy = self.model.evaluate(x_test, y_test)[1]
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        # per_class_accuracy = np.mean(y_test == np.round(self.model.predict(x_test)), axis=0)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        return accuracy, per_class_accuracy, cm

class RNNModel3:
    def __init__(self, x_train, num_classes=6):
        self.model = Sequential()
        self.model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=64))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_and_evaluate(self, x_train, x_test, y_train, y_test, batch_size=1000, epochs=10, validation_split=0.1):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
        accuracy = self.model.evaluate(x_test, y_test)[1]
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        # per_class_accuracy = np.mean(y_test == np.round(self.model.predict(x_test)), axis=0)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        return accuracy, per_class_accuracy, cm

def main():
    parser = argparse.ArgumentParser(description='Training LSTM models on the dataset for the classification task.')

    parser.add_argument('--dataset', default='data/moore_clean_cols_all.csv', type=str,
                        help='Path to the dataset file.')
    parser.add_argument('--test_size', default=0.2, type=float,
                        help='Size of the test set.')
    parser.add_argument('--validation_size', default=0.1, type=float,
                        help='Size of the validation set.')
    parser.add_argument('--num_classes', default=6, type=int,
                        help='Number of classes to keep in the dataset.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state for splitting the dataset.')
    parser.add_argument('--oversample', default=False, type=bool,
                        help='Oversample the dataset to solve the imbalance in the dataset')
    parser.add_argument('--selected_columns', default=[0, 1, 2, 3, 4, 5],
                        help='Oversample the dataset to solve the imbalance in the dataset')
    parser.add_argument('--num_rounds', default=20, type=int,
                        help='Number of rounds to train the model')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size for training the model')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Learning rate for training the model')
    parser.add_argument('--decay', default=0.9, type=float,
                        help='Decay rate for training the model')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for training the model')
    parser.add_argument('--percentage_poisoned', default=0.1, type=float,
                        help='Percentage of poisoned data in the client data')
    parser.add_argument('--attackers_percentage', default=0.2, type=float,
                        help='Percentage of attackers in the network')

    args = parser.parse_args()

    # Set the seed for NumPy
    np.random.seed(args.random_state)

    print(f'Training the model on the dataset: {args.dataset}')

    df = read_data(args.dataset)  # Load the dataset

    if args.selected_columns:
        df = df[df['266'].isin(args.selected_columns)]
    if args.oversample:
        df = oversample(df)
    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.num_classes, test_size=args.test_size,
                                                      random_state=args.random_state)
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    # TODO: handle the case where the selected columns are not in order for the categorical values

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Reshape the data and convert to one-hot encoding
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train = keras.utils.to_categorical(y_train, num_classes=args.num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=args.num_classes)

    print(f'The shape of the training set: {x_train.shape}')
    print(f'The shape of the testing set: {x_test.shape}')
    print(f'The shape of the training labels: {y_train.shape}')
    print(f'The shape of the testing labels: {y_test.shape}')

    # Create and evaluate model 1
    model_1 = RNNModel1(x_train)
    accuracy_1, per_class_accuracy_1, cm_1 = model_1.train_and_evaluate(x_train, x_test, y_train, y_test,
                                                                        batch_size=args.batch_size, epochs=args.epochs,
                                                                        validation_split=args.validation_size)
    print("Model 1 overall accuracy:", accuracy_1)
    print("Model 1 per class accuracy:", per_class_accuracy_1)
    print("Model 1 confusion matrix:")
    print(cm_1)

    # Create and evaluate model 2
    model_2 = RNNModel2(x_train)
    accuracy_2, per_class_accuracy_2, cm_2 = model_2.train_and_evaluate(x_train, x_test, y_train, y_test,
                                                                        batch_size=args.batch_size, epochs=args.epochs,
                                                                        validation_split=args.validation_size)

    print("Model 2 overall accuracy:", accuracy_2)
    print("Model 2 per class accuracy:", per_class_accuracy_2)
    print("Model 2 confusion matrix:")
    print(cm_2)

    # Create and evaluate model 3
    model_3 = RNNModel3(x_train)
    model_3.model.summary()
    accuracy_3, per_class_accuracy_3, cm_3 = model_3.train_and_evaluate(x_train, x_test, y_train, y_test,
                                                                        batch_size=args.batch_size, epochs=args.epochs,
                                                                        validation_split=args.validation_size)
    print("Model 3 overall accuracy:", accuracy_3)
    print("Model 3 per class accuracy:", per_class_accuracy_3)
    print("Model 3 confusion matrix:")
    print(cm_3)

    # Compare the performance of the models
    accuracies = [accuracy_1, accuracy_2, accuracy_3]
    best_model_index = np.argmax(accuracies)
    best_model = [model_1.model, model_2.model, model_3.model][best_model_index]
    best_accuracy = accuracies[best_model_index]
    best_per_class_accuracy = [per_class_accuracy_1, per_class_accuracy_2, per_class_accuracy_3][best_model_index]
    best_cm = [cm_1, cm_2, cm_3][best_model_index]

    print(f"The best model is model {best_model_index + 1}")
    print(f"The best model accuracy is {best_accuracy}")
    print(f"The best model per class accuracy is {best_per_class_accuracy}")
    print(f"The best model confusion matrix is {best_cm}")

    # Save the best model
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f"models/lstm_{date_time}.h5" # new version is model.save('my_model.keras')
    best_model.save(model_name)


if __name__ == '__main__':
    main()
