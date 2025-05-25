from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
from datetime import datetime
import argparse
import pandas as pd

from FL_genetic_attack import generate_trigger_updated, add_backdoor
from dataset_creating import (read_data, create_dataset,
                              generate_noniid_client_shards, generate_iid_client_data,
                              oversample, undersample, calculate_shards_and_rows)
from lstm_train import RNNModel1, RNNModel2, RNNModel3
from sklearn.metrics import confusion_matrix

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RNN models for FL with genetic attack')
    parser.add_argument('--path', type=str, default='data/Moore_6_classes.csv', help='Path to the dataset file')
    parser.add_argument('--sampling_method', default='none', type=str, help='Sampling method for the dataset', choices=['none', 'oversample', 'undersample'])
    parser.add_argument('--top_n', type=int, default=6, help='Number of classes to keep in the dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of FL rounds')
    parser.add_argument('--train_batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--attack_batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--train_epochs', type=int, default=1, help='Epochs per round')
    parser.add_argument('--attack_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--no_trials', type=int, default=5, help='Number of trials in the experiment')
    parser.add_argument('--num_attackers', type=int, default=1, help='Number of attackers in the experiment')
    parser.add_argument('--target', type=int, default=0, help='Target class for the attack')
    parser.add_argument('--type_of_distribution', type=str, default='iid', help='Type of data distribution for clients')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for output files')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory for saved models')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    df = read_data(args.path)
    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    if args.sampling_method == 'oversample':
        df = oversample(df)
    elif args.sampling_method == 'undersample':
        df = undersample(df)


    print(f"Training the model on the dataset: {args.path}")
    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.top_n, test_size=args.test_size, random_state=args.random_state)



    for model_name, ModelClass in zip(['rnn_1', 'rnn_2', 'rnn_3'], [RNNModel1, RNNModel2, RNNModel3]):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_file = f"{args.output_dir}/Exp_{model_name}_{timestamp}.txt"

        with open(results_file, "w") as f:
            f.write(f"Experiment Setting\n1. One attacker\n2. Genetic trigger\n3. Attack one and check the resiliency\n4. Model {model_name}\n5. Attackers: {args.num_attackers}\n\n")

            model = ModelClass(x_train, args.top_n).model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary(print_fn=lambda x: f.write(x + '\n'))

            model_path = f"{args.models_dir}/rnn_clean_{model_name}.h5"
            if os.path.exists(model_path):
                print(f"Loading pretrained model: {model_name}")
                model.load_weights(model_path)
            else:
                model.fit(x_train, y_train, validation_split=0.1, epochs=args.train_epochs, batch_size=args.train_batch_size, verbose=1)
                model.save(model_path)

            _, base_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f"Base Accuracy: {base_acc*100:.2f}%")

            if args.type_of_distribution == 'noniid':
                num_shards, shard_size = calculate_shards_and_rows(len(x_train))
                client_x, client_y = generate_noniid_client_shards(x_train, y_train, num_shards=num_shards, shard_size=shard_size, num_clients=args.num_clients, min_shard=1, max_shard=30)
                num_shards, shard_size = calculate_shards_and_rows(len(x_test))
                client_x_test, client_y_test = generate_noniid_client_shards(x_test, y_test, num_shards=num_shards, shard_size=shard_size, num_clients=args.num_clients, min_shard=1, max_shard=30)
            else:
                client_x, client_y = generate_iid_client_data(x_train, y_train, num_clients=args.num_clients)
                client_x_test, client_y_test = generate_iid_client_data(x_test, y_test, num_clients=args.num_clients)

            trials_accuracy = []
            trials_asr = []

            for trial in range(args.no_trials):
                print(f"Trial {trial+1}")
                f.write(f"Trial {trial+1}\n")

                models = [keras.models.clone_model(model) for _ in range(args.num_clients)]
                for m in models:
                    m.set_weights(model.get_weights())
                    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                test_triggers = {}

                for rnd in range(args.num_rounds):
                    print(f"Round {rnd+1}")
                    selected = random.sample(range(args.num_clients), min(args.num_clients, 3))
                    weights = []

                    for idx, client_id in enumerate(selected):
                        m = models[client_id]
                        if rnd == 1 and idx < args.num_attackers:
                            trigger, lox = generate_trigger_updated(client_x[client_id], client_y[client_id], client_x_test[client_id], client_y_test[client_id], model, m, idx)
                            benign_x, poison_x, benign_y, poison_y = add_backdoor(client_x[client_id], client_y[client_id], 0.2, lox, trigger, args.target)
                            x = np.concatenate([benign_x, poison_x])
                            y = np.concatenate([benign_y, poison_y])
                            m.fit(x, y, epochs=args.attack_epochs, batch_size=args.attack_batch_size, verbose=0)
                            test_triggers[client_id] = (poison_x, poison_y)
                        else:
                            m.fit(client_x[client_id], client_y[client_id], epochs=args.attack_epochs, batch_size=args.attack_batch_size, verbose=0)
                        weights.append(m.get_weights())

                    new_weights = [np.mean(np.array(w), axis=0) for w in zip(*weights)]
                    for m in models:
                        m.set_weights(new_weights)
                    model.set_weights(new_weights)

                _, acc = model.evaluate(x_test, y_test, verbose=0)
                trials_accuracy.append(acc)
                f.write(f"Final Accuracy: {acc:.4f}\n")

                round_asrs = []
                for cid, (tx, ty) in test_triggers.items():
                    _, asr = model.evaluate(tx, ty, verbose=0)
                    round_asrs.append(asr)
                    f.write(f"ASR for client {cid}: {asr:.4f}\n")
                avg_asr = np.mean(round_asrs) if round_asrs else 0
                trials_asr.append(avg_asr)
                f.write(f"Average ASR: {avg_asr:.4f}\n")

            summary_df = pd.DataFrame({'accuracy': trials_accuracy, 'ASR': trials_asr})
            summary_path = f"{args.output_dir}/summary_{model_name}_{timestamp}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
    print("Experiment completed successfully.")
