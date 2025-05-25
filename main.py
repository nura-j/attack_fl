import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
from datetime import datetime
import argparse
from dataset_creating import read_data, create_dataset, generate_noniid_client_shards, generate_iid_client_data
from lstm_train import RNNModel1, RNNModel2, RNNModel3
from sklearn.metrics import confusion_matrix

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RNN models for FL with genetic attack')
    parser.add_argument('--path', type=str, default='data/Moore_6_classes.csv', help='Path to the dataset file')
    parser.add_argument('--top_n', type=int, default=6, help='Number of classes to keep in the dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of FL rounds')
    parser.add_argument('--train_batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--attack_batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--train_epochs', type=int, default=10, help='Epochs per round')
    parser.add_argument('--attack_epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--type_of_distribution', type=str, default='iid', help='Type of data distribution for clients')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for output files')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory for saved models')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # Load data - assuming these functions exist elsewhere in your code
    # Replace with your actual data loading logic
    df = read_data(args.path)
    del df['Unnamed: 0']  # Remove unnecessary column if it exists
    print(f'Training the model on the dataset: {args.path}')
    print(f'The shape of the dataset: {df.head()}')
    print(f'The number of classes in the dataset: {len(df["class"].unique())}')
    x_training_data, x_test_data, y_training_data, y_test_data = create_dataset(df, top_n=args.top_n, test_size=args.test_size, random_state=args.random_state)
    print(f'The shape of the training set: {x_training_data.shape}')
    print(f'The shape of the testing set: {x_test_data.shape}')
    print(f'The shape of the training labels: {y_training_data.shape}')
    print(f'The shape of the testing labels: {y_test_data.shape}')
    exit(1)

    # Test each model
    for model_name in ['rnn_1', 'rnn_2', 'rnn_3']:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_file = f"{args.output_dir}/Exp_{model_name}_{timestamp}.txt"

        with open(results_file, "w") as f:
            f.write(f"Experiment Setting\n"
                    f"1. One attacker\n"
                    f"2. Genetic trigger\n"
                    f"3. Attack one and check the resiliency of the attack\n"
                    f"4. Model {model_name}\n"
                    f"5. Percentage of attackers 10%\n\n")

            # Initialize model
            if model_name == 'rnn_1':
                model = RNNModel1(x_training_data, args.top_n).model
            elif model_name == 'rnn_2':
                model = RNNModel2(x_training_data, args.top_n).model
            elif model_name == 'rnn_3':
                model = RNNModel3(x_training_data, args.top_n).model

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # Print model summary to file
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n")
            print(f"Model {model_name} initialized and compiled.")
            print(f"Model summary written to {results_file}")
            print(model.summary())
            # check if model already exists - load it, otherwise train a new one
            if os.path.exists(f"{args.models_dir}/rnn_clean_{model_name}.h5"):
                print(f"Loading pretrained model: {model_name}")
                model.load_weights(f"{args.models_dir}/rnn_clean_{model_name}.h5")
            else:
                print(f"No pretrained model found for {model_name}, training a new one.")
                # Train the model on the full dataset
                clean_his = model.fit(x_training_data, y_training_data, validation_split=0.1, epochs = args.train_epochs, batch_size = args.train_batch_size, verbose=1)
                model.save(f"{args.models_dir}/rnn_clean_{model_name}.h5")

            scores = model.evaluate(x_test_data, y_test_data)
            LSTM_accuracy = scores[1] * 100
            print('Test accuracy: ', scores[1] * 100, '%')

            y_pred = model.predict(x_test_data)
            y_pred = (y_pred > 0.5)
            matrix = confusion_matrix(y_test_data.argmax(axis=1), y_pred.argmax(axis=1))
            print(matrix)

            per_class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
            sum = 0
            for i in range(len(per_class_accuracy)):
                print("class {} accuracy: {}".format(i, per_class_accuracy[i]))
                sum += per_class_accuracy[i]
            print("Average accuracy: {}".format(sum / len(per_class_accuracy)))

            if args.type_of_distribution == 'noniid':
                print("Non-IID distribution selected.")
                # client_x, client_y, client_x_test, client_y_test = generate_noniid_client_shards(X_train, y_train, num_shards, shard_size, num_clients=10,
                #                   min_shard=1, max_shard=30) #todo shred_size
            else:
                client_x, client_y = generate_iid_client_data(x_training_data, y_training_data, num_clients=args.num_clients)
                client_x_test, client_y_test = generate_iid_client_data(x_test_data, y_test_data , num_clients=args.num_clients)

                print(f"Generated IID data for {args.num_clients} clients.")

            from FL_genetic_attack import calc_CAD, calc_ASR, mutation, crossover_genomes, crossover, add_backdoor, calc_fitness_4, generate_trigger
            no_features = 216
            worst = [*range(20, 30, 1)]
            ratio = 0.2
            lox = [0, 1, 2]
            trigger = [99, 1, 4]
            target = 0
            client_X_m_1_test = []
            client_Y_m_1_test = []
            client_X_test = []
            client_Y_test = []
            for i in range(len(client_x)):
                A, X, B, Y = add_backdoor(client_x_test[i], client_y_test[i], ratio, lox, trigger, target)
                client_X_test.append(A)
                client_X_m_1_test.append(X)
                client_Y_m_1_test.append(Y)
                client_Y_test.append(B)
            print(f"Client data prepared for {args.num_clients} clients with backdoor attack.")
            client_X_m_1 = []
            client_Y_m_1 = []
            client_X = []
            client_Y = []

            for i in range(len(client_x)):
                A, X, B, Y = add_backdoor(client_x[i], client_y[i], ratio, lox, trigger, target)
                client_X_m_1.append(X)
                client_Y_m_1.append(Y)
                client_X.append(A)
                client_Y.append(B)
            print(f"Client data prepared for {args.num_clients} clients with backdoor attack.")

            _, test_m_x_1, _, text_m_y_1 = add_backdoor(x_test_data, y_test_data, ratio, lox, trigger, target)

            round = 0
            no_trails = 5
            traials_average_global_accuracy = []
            traials_average_global_attack_success = []
            num_attackers = 1
            for j in range(no_trails):
                round = j + 1
                print("Experiment no. " + str(j + 1) + "\n")
                f.write("Experiment no. " + str(j + 1) + "\n")
                f.write("-" * 50)

                models_acc_rate = [[] for i in range(args.num_clients)]
                models_attack_rate_1 = [[] for i in range(args.num_clients)]

                model.load_weights(f'models/rnn_clean_{model_name}.h5')  # Load the clean model
                models = [keras.models.clone_model(model) for i in range(args.num_clients)]
                models_m = [keras.models.clone_model(model) for i in range(args.num_clients)]
                # loading the initial weights for clean models
                for m in models:
                    m.set_weights(model.get_weights())
                    m.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

                # loading the initial weights for malcious models - not poisoned yet
                for m in models_m:
                    m.set_weights(model.get_weights())
                    m.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

                _, acc = model.evaluate(x_test_data, y_test_data, verbose=0)
                print("Global model accuracy on clean data: ", acc * 100)
                f.write("Global model accuracy on clean data: " + str(acc * 100) + "\n")

                no_participants = 2
                round_selected = 5
                no_malicious = 1  # 20%
                ratio = 0.2
                target = 10  # yt="WWW";
                accuracy_rate = []
                attack_success_rate_1 = []
                f.write("Training the clients\n")
                print("Training the clients")
                # This can be made as lists, but got several issues - to be cleaned
                # test_m_x_1 = 0
                # text_m_y_1 = 0

                # test_m_x_2 = 0
                # text_m_y_2 = 0
                num_rounds = 20
                flag = False
                for i in range(num_rounds):
                    print("Round: " + str(i + 1))
                    f.write("Round: " + str(i + 1) + "\n")
                    weights = []
                    indicies = random.sample(range(args.num_clients), no_participants)
                    print("Clients in the round ", indicies)
                    f.write("Clients in the round " + str(indicies) + "\n")

                    for n, c in enumerate(indicies):
                        m = models[c]
                        malcious_model = models_m[c]

                        if (i + 1) % round_selected == 0 and n < no_malicious:
                            flag = True
                            x = np.concatenate((client_X[c], client_X_m_1[c]))
                            y = np.concatenate((client_Y[c], client_Y_m_1[c]))

                            # m.fit(benign_data_X_1, benign_data_y_1, epochs= epoch, batch_size=BATCH_SIZE) #training on both clean and mal data
                            # m.fit(temp_x_1, temp_y_1, epochs= epoch, batch_size=BATCH_SIZE) #training on both clean and mal data
                            m.fit(x, y, epochs=args.attack_epoch, batch_size=args.attach_batch_size)  # training on both clean and mal data
                            models_m[c] = m  # replace the malicous model with the poinsed one
                            print("Poisoned model accuracy")
                            _, acc = m.evaluate(client_X_test[c], client_Y_test[c], verbose=0)
                            f.write("Malcious model accuracy on clean data: " + str(acc) + "\n")
                            print("Malcious model accuracy on clean data: ", acc * 100)
                            _, acc = m.evaluate(client_X_m_1_test[c], client_Y_m_1_test[c], verbose=0)  ##
                            print("Malcious Model  accuracy on poisned data: ", acc * 100)
                            f.write("Malcious model accuracy on poisoned data: " + str(acc) + "\n")


                        else:
                            m.fit(client_X[c], client_Y[c], epochs=args.attack_epoch, batch_size=args.attch_batch_size)
                        weights.append(m.get_weights())  ##should I average the new ones only or all
                    new_weights = list()
                    for weights_list_tuple in zip(*weights):
                        new_weights.append(
                            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                        )
                        ##setting the new weights for all clients
                    for m in models:
                        m.set_weights(new_weights)
                    model.set_weights(new_weights)
                    print("")

                    _, acc = model.evaluate(x_test_data, y_test_data, verbose=0)
                    print("Global model accuracy on clean data: ", acc * 100)
                    f.write("Global model accuracy on clean data: " + str(acc) + "\n")
                    accuracy_rate.append(acc)
                    if flag:
                        _, acc = model.evaluate(test_m_x_1, text_m_y_1, verbose=0)
                        print("Global model accuracy on poisned data trigger1: ", acc * 100)
                        f.write("Global model accuracy on poisoned data with trigger 1: " + str(acc) + "\n")
                        attack_success_rate_1.append(acc)

                        # _, acc = model.evaluate(test_m_x_2, text_m_y_2, verbose=0)
                        # print("Global model accuracy on poisned data trigger 2: ", acc*100)
                        # f.write("Global model accuracy on poisoned data with trigger 2: " + str(acc) + "\n")
                        # attack_success_rate_1.append(acc)

                # if j == 0:
                #   traials_average_global_accuracy = (accuracy_rate).copy()
                #   traials_average_global_attack_success = (attack_success_rate_1).copy()
                #   traials_average_m = (models_attack_rate_1).copy()
                #   traials_average_b = (models_acc_rate).copy()
                # else:
                #   print("adding to previus the weights")
                #   traials_average_global_accuracy = [(traials_average_global_accuracy[i] + accuracy_rate[i])  for i in range(len(accuracy_rate))]
                #   traials_average_global_attack_success = [(traials_average_global_attack_success[i] + attack_success_rate_1[i])  for i in range(len(attack_success_rate_1))]
                #   traials_average_m = [[(traials_average_m[i][j] + models_attack_rate_1[i][j])  for j in range (len(traials_average_m[0]))] for i in range(len(traials_average_m))]
                #   traials_average_b = [[(traials_average_b[i][j] + models_acc_rate[i][j])  for j in range (len(traials_average_b[0]))] for i in range(len(traials_average_b))]

                print("\n\n\n\n\n")
                f.write("-" * 50)
            attack_success_rate_1 = [(x / no_trails) for x in attack_success_rate_1]
            traials_average_global_accuracy = [(x / no_trails) for x in traials_average_global_accuracy]
            # traials_average_m = [(x/no_trails) for x in traials_average_m]
            f.close()
            # exit(1)

