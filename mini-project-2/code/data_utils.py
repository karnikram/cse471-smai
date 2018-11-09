import os
import pickle
import numpy as np

# Loads a specified batch from the cifar-10 dataset
def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')

    X = dict['data'].astype('float')
    Y = dict['labels']

    Y = np.array(Y)

    return X, Y

# Loads all the batches of the cifar-10 dataset and returns them as np arrays
def load_cifar10(path):
    Xs, Ys = [], []

    for i in range(1,6):
        filename = os.path.join(path, 'data_batch_{}'.format(i))
        X, Y = load_cifar10_batch(filename) 
        Xs.extend(X)
        Ys.extend(Y)

    del X, Y

    Xtr = np.array(Xs)
    Ytr = np.array(Ys)

    Xte, Yte = load_cifar10_batch(os.path.join(path, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
