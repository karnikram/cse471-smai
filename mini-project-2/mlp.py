import data_utils
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import time
from copy import deepcopy

plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("--mean_sub", help="Perform mean subtraction on the features", action="store_true")
parser.add_argument("--scaling", help="Scale features to [0 1] range", action="store_true")
parser.add_argument("--pca", help="Perform PCA on features", action="store_true")
parser.add_argument("--lda", help="Perform LDA on features", action="store_true")

args = parser.parse_args()

# Load data
Xtr, Ytr, Xte, Yte = data_utils.load_cifar10('./cifar-10-batches-py')
num_train_samples = Xtr.shape[0]
num_test_samples = Xte.shape[0]

# Small development set for quick training and testing
num_dev_samples = 1000
mask = np.random.choice(num_train_samples, num_dev_samples, replace=False)
Xdev_tr = Xtr[mask]
Ydev_tr = Ytr[mask]

mask = np.random.choice(num_test_samples, int(num_dev_samples/2), replace=False)
Xdev_te = Xte[mask]
Ydev_te = Yte[mask]

# Unit scaling and mean subtraction
if(args.scaling):
    Xtr_dev = Xtr_dev/255
    Xte = Xte_dev/255

if(args.mean_sub):
    mean_image = np.mean(Xtr_dev,axis=0)
    Xtr_dev = Xtr_dev - mean_image
    Xte_dev = Xte_dev - mean_image

if(args.pca):
    pca = PCA(n_components=32)
    pca.fit(Xtr_dev)
    pca.transform(Xtr)
    pca.transform(Xte)

if(args.lda):
    lda = LDA(n_components=9) #TODO
    lda.fit(Xtr,Ytr)
    lda.transform(Xtr)
    lda.transform(Xte)

# Sanity checks
print('Shape of train data: {}'.format(Xtr.shape))
print('Shape of train labels: {}'.format(Ytr.shape))
print('Shape of test data: {}'.format(Xte.shape))
print('Shape of test labels: {}'.format(Yte.shape))

# Mult-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(1500,))
mlp.fit(Xdev_tr,Ydev_tr)

print('Train set accuracy of mlp: {}'.format(mlp.score(Xdev_tr,Ydev_tr)))
print('Test set accuracy of mlp: {}'.format(mlp.score(Xte,Yte)))
