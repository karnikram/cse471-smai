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
parser.add_argument("--best", help="Train mlp classifier with pre-determined parameters and report test set accuracy", action="store_true")
parser.add_argument("--search_lu", help="Perform 5-fold cross-validation for searching hidden layer dimensions", action="store_true")
parser.add_argument("--search_lr", action="store_true")
parser.add_argument("--mean_sub", help="Perform mean subtraction on the features", action="store_true")
parser.add_argument("--scaling", help="Scale features to [0 1] range", action="store_true")
parser.add_argument("--pca", help="Perform PCA on features", action="store_true")
parser.add_argument("--lda", help="Perform LDA on features", action="store_true")

args = parser.parse_args()

# Load data
Xtr, Ytr, Xte, Yte = data_utils.load_cifar10('./cifar-10-batches-py')
num_train_samples = Xtr.shape[0]
num_test_samples = Xte.shape[0]

# Unit scaling and mean subtraction
if(args.scaling):
    Xtr = Xtr/255
    Xte = Xte/255

if(args.mean_sub):
    mean_image = np.mean(Xtr,axis=0)
    Xtr = Xtr - mean_image
    Xte = Xte - mean_image

if(args.pca):
    pca = PCA(n_components=32)
    pca.fit(Xtr)
    pca.transform(Xtr)
    pca.transform(Xte)

if(args.lda):
    lda = LDA(n_components=9) #TODO
    lda.fit(Xtr,Ytr)
    lda.transform(Xtr)
    lda.transform(Xte)

# Small development set for quick training and testing
num_dev_samples = 500
mask = np.random.choice(num_train_samples, num_dev_samples, replace=False)
Xtr_dev = Xtr[mask]
Ytr_dev = Ytr[mask]

mask = np.random.choice(num_test_samples, int(num_dev_samples/2), replace=False)
Xte_dev = Xte[mask]
Yte_dev = Yte[mask]

num_folds = 5
Xtr_folds = np.split(Xtr_dev,num_folds)
Ytr_folds = np.split(Ytr_dev,num_folds)

# Sanity checks
print('Shape of train data: {}'.format(Xtr.shape))
print('Shape of train labels: {}'.format(Ytr.shape))
print('Shape of test data: {}'.format(Xte.shape))
print('Shape of test labels: {}'.format(Yte.shape))

del Xtr,Ytr

# 5-fold cross-validation for number of hidden layers and units

if(args.search_lu):

    mlp = MLPClassifier()
    best_mlp = None
    best_mean_accuracy = 0

    lu_to_accuracies = {}
    u_choices = [50,100,500,1000]
    l_choices = [1,2,3,4]

    for u in u_choices:
        for l in l_choices:
            mlp.set_params(hidden_layer_sizes=(u,) * l)
            lu_accuracy = np.zeros(num_folds)
            for i in range(num_folds):
                tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
                tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
                val_data = Xtr_folds[i]
                val_labels = Ytr_folds[i]
                mlp.fit(tr_data,tr_labels)
                lu_accuracy[i] = mlp.score(val_data,val_labels)

            if(np.mean(lu_accuracy) > best_mean_accuracy):
                best_l = l
                best_u = u
                best_mlp = mlp
                best_mean_accuracy = np.mean(lu_accuracy)
            lu_to_accuracies[(l,u)] = lu_accuracy

    for l,u in sorted(lu_to_accuracies):
            print('l = {}, u = {}, mean_accuracy = {}'.format(l,u,np.mean(lu_to_accuracies[(l,u)])))

    print('Best choice for number of layers, units per layer from cross-validation: {}, {}'.format(best_l,best_u))

# 5-fold cross-validation for choice of learning rate
if(args.search_lr):

    mlp = MLPClassifier()
    best_mlp = None
    best_mean_accuracy = 0

    best_l = 3
    best_u = 500
    lr_to_accuracies = {}
    lr_choices = [0.001,0.01,0.1,1]

    for lr in lr_choices:
        mlp.set_params(learning_rate_init=lr,hidden_layer_sizes=((best_u,)*best_l))
        lr_accuracy = np.zeros(num_folds)
        for i in range(num_folds):
            tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
            tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
            val_data = Xtr_folds[i]
            val_labels = Ytr_folds[i]
            mlp.fit(tr_data,tr_labels)
            lr_accuracy[i] = mlp.score(val_data,val_labels)

        if(np.mean(lr_accuracy) > best_mean_accuracy):
            best_mean_accuracy = np.mean(lr_accuracy)
            best_lr = lr
            best_mlp = mlp
            lr_to_accuracies[lr] = lr_accuracy

    for lr in sorted(lr_to_accuracies):
            print('lr = {}, mean_accuracy = {}'.format(lr,np.mean(lr_to_accuracies[lr])))

    for lr in lr_choices:
        accuracies = lr_to_accuracies[lr]
        plt.scatter([lr]*len(accuracies),accuracies)

    mean_accuracies = np.array([np.mean(a) for lr,a in sorted(lr_to_accuracies.items())])
    std_accuracies = np.array([np.std(a) for lr,a in sorted(lr_to_accuracies.items())])
    plt.errorbar(lr_choices,mean_accuracies,yerr=std_accuracies)
    plt.title('Cross-validation on learning rate')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.savefig('/plots/mlp-lr.png',bbox_inches='bbox')

    print('Best choice for learning rate from cross-validation: {}'.format(best_lr))

if(args.best):

    best_mlp = MLPClassifier()
    best_l = 3
    best_u = 500
    best_lr = 0.0001
    best_mlp.set_params(hidden_layer_sizes=(best_u,) * best_l,learning_rate_init=best_lr)
    print('Test set accuracy of best mlp: {}'.format(best_mlp.score(Xte,Yte)))
