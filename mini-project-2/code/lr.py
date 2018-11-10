import data_utils
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt

plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("--best", help="Train LR classifier with pre-determined parameters and report test set accuracy", action="store_true")
parser.add_argument("--search_c", help="Search for choice of regularization factor", action="store_true")
parser.add_argument("--mean_sub", help="Perform mean subtraction on the features", action="store_true")
parser.add_argument("--scaling", help="Scale features to [0 1] range", action="store_true")
parser.add_argument("--pca", help="Perform PCA on features", action="store_true")
parser.add_argument("--lda", help="Perform LDA on features", action="store_true")

args = parser.parse_args()

# Load dataset
Xtr, Ytr, Xte, Yte = data_utils.load_cifar10('../cifar-10-batches-py')
num_train_samples = Xtr.shape[0]
num_test_samples = Xte.shape[0]

# Unit scaling and mean subtraction
if(args.scaling):
    print('Performing feature scaling to [0 1]')
    Xtr = Xtr/255
    Xte = Xte/255

if(args.mean_sub):
    print('Performing mean subtraction on the samples')
    mean_image = np.mean(Xtr,axis=0)
    Xtr = Xtr - mean_image
    Xte = Xte - mean_image

if(args.pca):
    print('Performing PCA on the samples')
    pca = PCA(n_components=0.9)
    pca.fit(Xtr)
    print('Number of components used: {}'.format(pca.n_components_))
    Xtr = pca.transform(Xtr)
    Xte = pca.transform(Xte)

if(args.lda):
    print('Performing LDA on the samples')
    lda = LDA()
    lda.fit(Xtr,Ytr)
    print('Number of components used: {}'.format(lda.explained_variance_ratio_.shape))
    Xtr = lda.transform(Xtr)
    Xte = lda.transform(Xte)

# Small development set for quick hyperparameter search
num_dev_samples = 5000
np.random.seed(28)
mask = np.random.choice(num_train_samples, num_dev_samples, replace=False)
Xtr_dev = Xtr[mask]
Ytr_dev = Ytr[mask]

np.random.seed(28)
mask = np.random.choice(num_test_samples, int(num_dev_samples/5), replace=False)
Xte_dev = Xte[mask]
Yte_dev = Yte[mask]

num_folds = 5
Xtr_folds = np.split(Xtr_dev,num_folds)
Ytr_folds = np.split(Ytr_dev,num_folds)

# 5-fold cross-validation for choosing hyperparameter C
if(args.search_c):

    print('Running hyperparamer search with cross-validation for optimal C')
    print('Shape of train data: {}'.format(Xtr_dev.shape))
    print('Shape of train labels: {}'.format(Ytr_dev.shape))
    lr = LogisticRegression(solver='lbfgs',multi_class='multinomial')
    best_mean_accuracy = 0

    #c_choices = [0.1, 1, 10, 100, 1000]
    #c_choices = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    c_choices = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    c_to_accuracies = {}
   
    for c in c_choices:
        lr.set_params(C=c)
        c_accuracy = np.zeros(num_folds)
        for i in range(num_folds):
            tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
            tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
            val_data = Xtr_folds[i]
            val_labels = Ytr_folds[i]
            lr.fit(tr_data,tr_labels)
            c_accuracy[i] = lr.score(val_data,val_labels)

        if(np.mean(c_accuracy) >= best_mean_accuracy):
            best_mean_accuracy = np.mean(c_accuracy)
            best_c = c
        c_to_accuracies[c] = c_accuracy

    for c in sorted(c_to_accuracies):
        print('c = {}, mean_accuracy = {}'.format(c,np.around(np.mean(c_to_accuracies[c]),4)))

    for c in c_choices:
        accuracies = c_to_accuracies[c]
        plt.scatter([c]*len(accuracies),accuracies)

    mean_accuracies = np.array([np.mean(a) for c,a in sorted(c_to_accuracies.items())])
    std_accuracies = np.array([np.std(a) for c,a in sorted(c_to_accuracies.items())])
    plt.errorbar(c_choices,mean_accuracies,yerr=std_accuracies)
    plt.xscale('log')
    plt.title('Cross-validation on C')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.savefig('../plots/lreg-c.png',bbox_inches='tight')

    print('Best choice for C from cross-validation: {}'.format(best_c))

if(args.best):

    print('Training LR with best choice of C')
    print('Shape of train data: {}'.format(Xtr.shape))
    print('Shape of train labels: {}'.format(Ytr.shape))
    print('Shape of test data: {}'.format(Xte.shape))
    print('Shape of test labels: {}'.format(Yte.shape))

    best_lr = LogisticRegression(solver='lbfgs',multi_class='multinomial')
    best_c = 0.0000001

    print('C = {}'.format(best_c))

    best_lr.set_params(C=best_c)
    best_lr.fit(Xtr,Ytr)
    Yte_pred = best_lr.predict(Xte)
    print('Test set accuracy of best LR: {}'.format(accuracy_score(Yte,Yte_pred)))
    print('Test set f1-score of best LR: {}'.format(f1_score(Yte,Yte_pred,average='macro')))
