import argparse
import random
import numpy as np
from numpy.random import randint, rand
from sklearn.model_selection import train_test_split
import keras
from random import sample
from dataset_creating import read_data, create_dataset, oversample, undersample
from dataset_creating import generate_iid_client_data, generate_noniid_client_shards, calculate_shards_and_rows, \
    calculate_shards_and_rows_simple

from lstm_train import RNNModel1, RNNModel2, RNNModel3


#  calculating the CAD
def calc_CAD(net1, net2, X, y):
    _, scores_1 = net1.evaluate(X, y)  # calculate the accuracy of the first network
    _, scores_2 = net2.evaluate(X, y)  # calculate the accuracy of the second network
    cad = np.abs(scores_1 - scores_2)  # calculate the abs difference between them
    # print("CAD: ", cad)
    return cad


# calculate ASR
def calc_ASR(model, X, y):
    _, scores_1 = model.evaluate(X, y)
    # print("ASR: ", scores_1)
    return np.abs(scores_1)


# mutation operator
def mutation(bitstring):
    r_mut = 1 / len(bitstring)
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() > r_mut:
            bitstring[i] = 1 - bitstring[i]
    return bitstring


# crossover two parents to create two children
def crossover_genomes(p1, p2, r_cross=0.6):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return c1, c2


# crossover two parents to create two children
def crossover(parents):
    L = len(parents)
    for i in range(0, len(parents) - 2, 2):
        parents[i], parents[i + 1] = crossover_genomes(parents[i], parents[i + 1])
    return parents


def add_backdoor(x_test, y_test, ratio, lox, trigger, target, num_classes=6):
    per = ratio / 100
    benign_data_x, mal_data_x, benign_data_y, mal_data_y = train_test_split(x_test, y_test,
                                                                            test_size=per)

    for i, item in enumerate(mal_data_x):  # add the trigger for the poisoned part
        for x, y in zip(lox, trigger):
            item[x] = y
        mal_data_x[i] = item
        mal_data_y[:] = keras.utils.to_categorical(target, num_classes=num_classes)

    return benign_data_x, mal_data_x, benign_data_y, mal_data_y


def generate_trigger(x_train, y_train, x_test, y_test, clean_model, mal_model, worst):
    pop_size = 20
    n_bits = 48
    n_pop = pop_size
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    parents = pop
    n_iter = 2  # 100 simulation or 10~5 with confidence interval
    best_fit_arr = []
    mean_fit_arr = []
    for i in range(n_iter):
        sum_fit = 0
        max_fit = 0
        Trigger = [0, 0, 0]
        Lox = [0, 0, 0]
        fit_arr = dict()

        os = crossover(parents)  # crossover

        os = [list(mutation(np.array(ind))) for ind in os]  # mutation
        combined = parents + os

        for j in range(len(combined)):
            fit, trigger, lox = calc_fitness_4(x_train, y_train, x_test, y_test, clean_model, mal_model, combined[j])
            sum_fit += fit
            fit_arr[j] = fit
            if fit >= max_fit:  # find the trigger with best fitness
                Trigger = trigger
                Lox = lox
                max_fit = fit

        d = sorted(fit_arr.items(), key=lambda x: x[1], reverse=True)

        bad_others_lox = random.sample(worst, 5)

        parents = [combined[d[i][0]] for i in range(15)]
        for a in bad_others_lox:
            parents.append(combined[a])
        best_fit_arr.append(max_fit)
        mean_fit_arr.append(sum_fit / len(fit_arr))

        for k, item in enumerate(Lox):
            if item >= 216:  # number of features
                Lox[k] = Lox[k] - 1

    return Trigger, Lox


def calc_fitness_4(x_training_data, y_training_data, x_test_data, y_test_data, rnn_clean_2, rnn_m, genome,
                   no_features=216, alpha=.5, y_target=0, ratio=20):
    # XTrain = x_training_data  # Data used in creating the trigger using GA
    # YTrain = y_training_data  # Data used in creating the trigger using GA

    loc1 = genome[0:8]
    loc2 = genome[8:16]
    loc3 = genome[16:24]

    loc1 = ''.join([str(elem) for elem in loc1])
    x = int(str(loc1), 2)
    if x >= no_features:
        x = no_features - 1  # to avoid incorrect number of features

    loc2 = ''.join([str(elem) for elem in loc2])  # convert the list of bin to one number
    y = int(str(loc2), 2)
    if y >= no_features:
        y = no_features - 1  # check the zero case - edges

    loc3 = ''.join([str(elem) for elem in loc3])
    z = int(str(loc3), 2)
    if z >= no_features:
        z = no_features - 1  # to avoid incorrect number of features

    lox = [x, y, z]  # aka location of the trigger

    a = genome[24:32]  # Value of the trigger
    b = genome[32:40]
    c = genome[40:48]

    a = ''.join([str(elem) for elem in a])
    aa = int(str(a), 2)

    b = ''.join([str(elem) for elem in b])

    bb = int(str(b), 2)

    c = ''.join([str(elem) for elem in c])

    cc = int(str(c), 2)

    trigger = [aa, bb, cc]

    print("Calling the add backdoor")
    benign_data_x, mal_data_x, benign_data_y_one_hot, mal_data_y_one_hot = add_backdoor(x_training_data,
                                                                                        y_training_data, ratio, lox,
                                                                                        trigger, y_target)

    # Train the clean model
    his = rnn_clean_2.fit(benign_data_x, benign_data_y_one_hot, validation_split=0.3, epochs=1, batch_size=1000,
                          verbose=1)  # train on clean data (network 1 - clean model)

    # Concatenate the clean and poisoned data
    x = np.concatenate((benign_data_x, mal_data_x))  # combine clean and poisoned data
    y = np.concatenate((benign_data_y_one_hot, mal_data_y_one_hot))  # combine clean and poisoned data

    # Train the poisoned model
    his = rnn_m.fit(x, y, validation_split=0.3, epochs=1, batch_size=1000,
                    verbose=1)  # train on clean & poisoned data (network 2 - m model)

    print("Testing ASR")
    asr = calc_ASR(rnn_m, mal_data_x, mal_data_y_one_hot)

    print("Testing CAD")
    # print("x_test_data.shape: ", x_test_data.shape)
    # print("y_test_data.shape: ", y_test_data.shape)
    # print("y_test_data_one_hot.shape: ", y_test_data_one_hot.shape)
    cad = calc_CAD(rnn_clean_2, rnn_m, x_test_data,
                   y_test_data)  # NN: double check if it works with one hot encoded y or actual
    performance = alpha * asr - (1 - alpha) * abs(cad)
    return performance, trigger, lox


def genetic_attack(clients_x, clients_y, clients_x_test, clients_y_test, x_test, y_test,
                   model, model_path=None, output_path='results/', worst=None, ratio=20,
                   target=0, model_epoch=1, batch_size=100, num_rounds=10, no_participants=5,
                   number_attackers=1, attack_round=2):
    '''
    Genetic attack on the FL model
    :param worst: the selection from the worst performing clients
    :param clients_x:
    :param clients_y:
    :param clients_x_test:
    :param clients_y_test:
    :param x_test:
    :param y_test:
    :param num_rounds: number of round in the current FL pool
    :param no_participants: number of participants in the FL pool
    :param number_attackers: number of attackers in the FL pool
    :param attack_round: selected round of the attack
    :param model:
    :param model_path:
    :param output_path:
    :return:
    '''
    print('Genetic attack started!')
    model.summary()
    # Load initial clean-trained model
    if model_path:
        model.load_weights(model_path)
    model.summary()

    num_clients = len(clients_x)

    # Clone the model for clients
    clean_models = [keras.models.clone_model(model) for _ in range(num_clients)]
    malicious_models = [keras.models.clone_model(model) for _ in range(num_clients)]
    for i in range(num_clients):
        clean_models[i].set_weights(model.get_weights())
        clean_models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        malicious_models[i].set_weights(model.get_weights())
        malicious_models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Initial global model performance
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Initial model performance: %.3f' % (acc * 100))

    flag = False
    triggers_data = []
    for i in range(num_rounds):
        weights = []
        indices = random.sample(range(num_clients), no_participants)  # select clients of this round randomly
        attackers = random.sample(indices, number_attackers)  # select attackers randomly
        for n, c in enumerate(indices):
            m = clean_models[c]
            malicious_model = malicious_models[c]

            if i == attack_round and c in attackers:
                print('Attacking client: ', c)
                # Perform the genetic attack
                flag = True
                trigger, lox = generate_trigger(clients_x[c], clients_y[c], clients_x_test[c], clients_y_test[c], m,
                                                malicious_model, worst)
                # Add the backdoor to the model
                benign_data_x, mal_data_x, benign_data_y, mal_data_y = add_backdoor(clients_x[c], clients_y[c], ratio,
                                                                                    lox,
                                                                                    trigger, target)
                triggers_data.append([trigger, lox, mal_data_x, mal_data_y])
                # Train the model
                x = np.concatenate((benign_data_x, mal_data_x))
                y = np.concatenate((benign_data_y, mal_data_y))
                m.fit(x, y, epochs=model_epoch, batch_size=batch_size, verbose=1)
                malicious_models[c] = m  # replace the malicious model with the poisoned one
                # Evaluate the model
                _, acc = m.evaluate(clients_x_test[c], clients_y_test[c], verbose=0)
                print('Client %d accuracy on clean data: %.3f' % (c, acc * 100))
                _, acc = m.evaluate(mal_data_x, mal_data_y, verbose=0)
                print('Client %d accuracy on poisoned data: %.3f' % (c, acc * 100))
            else:
                m.fit(clients_x[c], clients_y[c], epochs=model_epoch, batch_size=batch_size, verbose=1)
            weights.append(m.get_weights())
        # Update the global model
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        model.set_weights(new_weights)
        # Setting the new weights of the clients to the global model
        for c in indices:
            clean_models[c].set_weights(model.get_weights())

        # Evaluate the global model
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Round %d, Global model performance: %.3f' % (i, acc * 100))

        # Testing the attack on the global model
        if flag:
            _, acc = model.evaluate(triggers_data[-1][-2], triggers_data[-1][-1], verbose=0)
            print('Global model performance after attack: %.3f' % acc * 100) # Note this trigger keeps changing
            flag = False

    return 0


def generate_trigger_updated(x_train, y_train, x_test, y_test, clean_model, mal_model, target):
    pop_size = 20  # pop_size=10;
    n_bits = 48
    n_pop = pop_size
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]  # pop_init = randi([0 1], pop_size,30);
    parents = pop  # parents=pop_init;
    # for indivisual in parents:
    #   print(indivisual)

    # size of population set it to fixed size (do not make it exponentioal)
    n_iter = 2  # 100 similation or 10~5 with confidence interval
    best_fit_arr = []
    mean_fit_arr = []
    for i in range(n_iter):
        print("Iteration: ", i)
        # print("len(parents)", len(parents))
        sum_fit = 0
        max_fit = 0
        Trigger = [0, 0, 0]
        Lox = [0, 0, 0]
        fit_arr = dict()
        # crossover
        os = crossover(parents)  # parent 10 - OS -5
        # mutation
        os = [list(mutation(np.array(ind))) for ind in os]
        combined = parents + os
        # print("len(combined)", len(combined))
        for j in range(len(combined)):  # Sample?
            # calc_fitness_4(x_training_data, y_training_data, x_test_data, y_test_data, rnn_clean_2, rnn_m, genome, no_features=no_features, alpha = .5)
            fit, trigger, lox = calc_fitness_4_updated(x_train, y_train, x_test, y_test, clean_model, mal_model,
                                                       combined[j], target)
            sum_fit += fit
            fit_arr[j] = fit
            if fit >= max_fit:
                Trigger = trigger
                Lox = lox
                max_fit = fit
            # max_fit = max(max_fit, fit) #store the trigger
        d = sorted(fit_arr.items(), key=lambda x: x[1], reverse=True)
        # per = np.random.permutation(pop_size)
        bad_others_lox = sample(worst, 5)
        # bad_others_lox = per[:3]#bad_others_lox= pop_size+randperm(pop_size,3);  #I will change this to rand
        parents = [combined[d[i][0]] for i in range(15)]  # Update 15 no the new value
        for i in bad_others_lox:
            parents.append(combined[i])
        best_fit_arr.append(max_fit)
        mean_fit_arr.append(sum_fit / len(fit_arr))
        print("LOX & TYPE", Lox, type(Lox))
        for i, item in enumerate(Lox):
            if item >= 216:  # number of features
                Lox[i] = Lox[i] - 1
    print("Trigger: ", Trigger)
    print("Lox: ", Lox)
    return Trigger, Lox


no_features = 216
worst = [*range(20, 30, 1)]

#Creating the poisoned data
#X_test -> the images
#Y_test -> the labels
#percentage -> the precentage of images to poison (ratio)
#attck_type -> a. single_class (targeted), b. multiple (non-targted)
#def add_backdoor(X_test, Y_test, trigger, percentage = 30, attck_type = "single_class", poison = "poison_1", target=5):
def add_backdoor_updated(X_test, Y_test, ratio, lox, trigger, target):
  per = ratio / 100
  benign_data_X, mal_data_X, benign_data_y, mal_data_Y = train_test_split(X_test, Y_test, test_size=per)
  for i, item in enumerate(mal_data_X):
    for x, y in zip(lox,trigger):
      item[x] = y
    mal_data_X[i] = item
    if target == 0:
      mal_data_Y[:] = [1., 0., 0., 0., 0., 0.]
    elif target == 1:
      mal_data_Y[:] = [0., 1., 0., 0., 0., 0.]
  #mal_data_Y[:] = target #automiate this later
  return benign_data_X, mal_data_X, benign_data_y, mal_data_Y

def calc_fitness_4_updated(x_training_data, y_training_data, x_test_data, y_test_data, rnn_clean_2, rnn_m, genome,
                           target, no_features=no_features,
                           alpha=.5):  # What is genome here? I assume it is one individual
    XTrain = x_training_data  # Sample - everytime
    YTrain = y_training_data

    ratio = 0.2  # ratio=0.2;
    loc1 = genome[0:8]  # loc1=genome(1:5);
    loc2 = genome[8:16]  # loc2=genome(6:10);
    loc3 = genome[
           16:24]  # loc3=genome(11:15); #2^8 -  256 but we have 211 features so we need to trim it (for now anything larger than 211 will be mapped to 211)

    # x=min(bin2dec(num2str(loc1)), 18); #What is happening here? + Why 18? 18 features?
    # x=max(x,axis = 1)#x=max(x,1); Is 1 the rows?
    loc1 = ''.join([str(elem) for elem in loc1])
    x = int(str(loc1), 2)
    if x >= no_features:
        x = no_features - 1  # check the zero case - edges

    # y=min(bin2dec(num2str(loc2)),18); #What is happening here?
    # y=max(y,axis = 1) #y=max(y,1);
    loc2 = ''.join([str(elem) for elem in loc2])  # convert the list of bin to one number
    y = int(str(loc2), 2)
    if y >= no_features:
        y = no_features - 1  # check the zero case - edges

    # z=min(bin2dec(num2str(loc3)),18); #What is happening here?
    # z=max(z,axis = 1) #z=max(z,1);
    loc3 = ''.join([str(elem) for elem in loc3])
    z = int(str(loc3), 2)
    if z >= no_features:
        z = no_features - 1  # check the zero case - edges

    lox = [x, y,
           z]  # lox=[x y z];aka location of the trigger (* WE MUST consider the case where x or y or z equal the same one)

    a = genome[24:32]  # a=genome(16:20); #Value of the trigger
    b = genome[32:40]  # b=genome(21:25);
    c = genome[40:48]  # c=genome(26:30);
    # aa=bin2dec(num2str(a)); #Here we need to fix the range (just to avoid any anomaly detect)
    a = ''.join([str(elem) for elem in a])
    aa = int(str(a), 2)
    # aa = fix_range(aa, x) #to normalize it

    b = ''.join([str(elem) for elem in b])
    # bb=bin2dec(num2str(b));
    bb = int(str(b), 2)
    # bb = fix_range(bb, y) #to normalize it

    c = ''.join([str(elem) for elem in c])
    # cc=bin2dec(num2str(c));
    cc = int(str(c), 2)
    # cc = fix_range(cc, z) #to normalize it (easy method define a dictionary of the features - keep the min and max of each one)

    trigger = [aa, bb, cc]  # trigger=[aa bb cc]; - make sure to fix the range

    y_target = 0  # yt="WWW"; -? Is it a good choice? (if the data balanced then no problem)

    # I assume this is the poised & bengin data?
    # [Xb,yb]=add_backdoor(XTrain,yTrain, ratio, lox, trigger, yt);
    # add_backdoor(X_test, Y_test, ratio, lox, trigger, target)
    print("Calling the add backdoor")
    benign_data_X, mal_data_X, benign_data_y_one_hot, mal_data_Y_one_hot = add_backdoor_updated(x_training_data,
                                                                                                y_training_data, .2,
                                                                                                lox, trigger, target)
    print("had finished the add backdoor")
    # benign_data_y_one_hot = to_categorical(benign_data_y, NUM_CLASSES)
    # mal_data_Y_one_hot = to_categorical(mal_data_Y, NUM_CLASSES)
    # Clean model training
    # net1 = trainNetwork(XTrain,yTrain,layers,options);

    print("Training clean model")
    print("benign_data_X.shape: ", benign_data_X.shape)
    print("benign_data_y_one_hot.shape: ", benign_data_y_one_hot.shape)
    print("mal_data_X.shape: ", mal_data_X.shape)
    print("mal_data_Y_one_hot.shape: ", mal_data_Y_one_hot.shape)
    his = rnn_clean_2.fit(benign_data_X, benign_data_y_one_hot, validation_split=0.3, epochs=1, batch_size=1000,
                          verbose=1)
    # net1 = trainNetwork(benign_data_X,benign_data_y)
    #
    x = np.concatenate((benign_data_X, mal_data_X))
    y = np.concatenate((benign_data_y_one_hot, mal_data_Y_one_hot))
    # N: THIS NEEDS UPDATE - I ONLY TRAIN ON M DATA
    his = rnn_m.fit(x, y, validation_split=0.3, epochs=1, batch_size=1000, verbose=1)
    # net2 = trainNetwork(Xb,yb,layers,options);

    # YPred = classify(net2,Xb);

    # asr=calc_ASR(YPred,yt,ratio);
    print("Testing ASR")
    asr = calc_ASR(rnn_m, mal_data_X, mal_data_Y_one_hot)
    # cad=calc_CAD(net1,net2,XTrain,yTrain);
    print("Testing CAD")
    cad = calc_CAD(rnn_clean_2, rnn_m, x_test_data, y_test_data)
    performance = (alpha) * asr - (1 - alpha) * abs(cad)  #
    return performance, trigger, lox


def main():
    parser = argparse.ArgumentParser(description='FL attack')
    parser.add_argument('--num_clients', default=100, type=int,
                        help='Number of clients to generate the data for')
    parser.add_argument('--dataset', default='data/moore_clean_cols_all.csv', type=str,
                        help='Path to the dataset file.')
    parser.add_argument('--selected_columns', default=[0, 1, 2, 3, 4, 5],
                        help='if we want to select specific columns from the dataset.')
    parser.add_argument('--sampling_method', default='none', choices=['oversample', 'undersample', 'none'],
                        help='Method to handle dataset imbalance: oversample, undersample, or none')
    parser.add_argument('--num_classes', default=6, type=int,
                        help='Number of classes to keep in the dataset.')
    parser.add_argument('--random_state', default=42, type=int,
                        help='Random state for splitting the dataset.')
    parser.add_argument('--test_size', default=0.2, type=float,
                        help='Size of the test dataset.')
    parser.add_argument('--type_of_distribution', default='non-iid', type=str,
                        help='Type of distribution to use for the clients data. Options are iid and non-iid.')
    parser.add_argument('--no_trials', default=2, type=int,
                        help='Number of trials to run the genetic algorithm.')
    parser.add_argument('--model_name', default='rnn_3', type=str,
                        help='Name of the model to use for the attack.')
    parser.add_argument('--num_rounds', default=10, type=int,
                        help='Number of rounds in the FL pool.')
    parser.add_argument('--no_participants', default=10, type=int,
                        help='Number of participants in the FL pool.')
    parser.add_argument('--number_attackers', default=2, type=int,
                        help='Number of attackers in the FL pool.')
    parser.add_argument('--attack_round', default=5, type=int,
                        help='Round to launch the attack.')
    parser.add_argument('--model_path', default='models/lstm_2024-06-17-19-25-38.h5', type=str,
                        help='Path to the model to use for the attack.')
    parser.add_argument('--output_path', default='results/', type=str,
                        help='Path to save the results of the attack.')

    args = parser.parse_args()

    df = read_data(args.dataset)  # Load the dataset

    if args.selected_columns:
        df = df[df['266'].isin(args.selected_columns)]

    if args.sampling_method == 'oversample':
        df = oversample(df)
    elif args.sampling_method == 'undersample':
        df = undersample(df)
    elif args.sampling_method == 'none':
        pass
    else:
        raise ValueError('Invalid sampling method. Please use either oversample, undersample, or none.')

    x_train, x_test, y_train, y_test = create_dataset(df, top_n=args.num_classes, test_size=args.test_size,
                                                      random_state=args.random_state)
    num_shards, shard_size = calculate_shards_and_rows(len(x_train))

    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    # TODO: handle the case where the selected columns are not in order for the categorical values

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train = keras.utils.to_categorical(y_train, num_classes=args.num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=args.num_classes)

    # TODO set the seeds
    worst = [*range(20, 30, 1)]  # TODO let this be auto based on values passed by user
    no_features = x_train.shape[1]

    if args.type_of_distribution == 'iid':
        clients_x, clients_y = generate_iid_client_data(x_train, y_train, args.num_clients)  ## IID dist
    elif args.type_of_distribution == 'non-iid':
        clients_x, clients_y = generate_noniid_client_shards(x_train, y_train, num_shards, shard_size)
    else:
        raise ValueError('Invalid type of distribution. Please use either iid or non-iid.')
    clients_x_test, clients_y_test = generate_iid_client_data(x_test, y_test, args.num_clients)  ## Non-IID dist

    # print('Client data generated successfully!')

    # Testing the adding backdoor function
    benign_data_x, mal_data_x, benign_data_y, mal_data_y = add_backdoor(x_test, y_test, 20, [2, 3, 4], [1, 1, 1], 4)

    # Load the intial model
    if args.model_name == 'rnn_1':
        model = RNNModel1(x_train)
    elif args.model_name == 'rnn_2':
        model = RNNModel2(x_train)
    elif args.model_name == 'rnn_3':
        model = RNNModel3(x_train)
    else:
        raise ValueError('Invalid model name. Please use either rnn_1, rnn_2, or rnn_3.')

