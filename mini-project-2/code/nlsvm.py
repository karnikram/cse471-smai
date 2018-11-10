import data_utils
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt

plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("--best", help="Train svm classifier with pre-determined parameters and report test set accuracy", action="store_true")
parser.add_argument("--search_cg", help="Search for choice of C and gamma", action="store_true")
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

# 5-fold cross-validation for choosing C and gamma
if(args.search_cg):

    print('Running grid search with cross-validation for optimal C and gamma')
    print('Shape of train data: {}'.format(Xtr_dev.shape))
    print('Shape of train labels: {}'.format(Ytr_dev.shape))

    svm = SVC()
    best_mean_accuracy = 0

    #c_choices = [0.0001, 0.01, 0.1, 1, 10]
    c_choices = [1, 10]
    #c_choices = [0.1, 1, 2, 4, 8]
    #g_choices = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    g_choices = [0.1, 1]

    cg_to_accuracies = {}

    for c in c_choices:
        for g in g_choices:
            svm.set_params(C=c,gamma=g)
            cg_accuracy = np.zeros(num_folds)
            for i in range(num_folds):
                tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
                tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
                val_data = Xtr_folds[i]
                val_labels = Ytr_folds[i]
                svm.fit(tr_data,tr_labels)
                cg_accuracy[i] = svm.score(val_data,val_labels)

            if(np.mean(cg_accuracy) > best_mean_accuracy):
                best_mean_accuracy = np.mean(cg_accuracy)
                best_c = c
                best_g = g
            cg_to_accuracies[(c,g)] = cg_accuracy

    for c,g in sorted(cg_to_accuracies):
            print('c = {}, g = {}, mean_accuracy = {}'.format(c,g,np.around(np.mean(cg_to_accuracies[(c,g)]),4)))

    print('Best choice for C and gamma from cross-validation: {}, {}'.format(best_c,best_g))

if(args.best):

    print('Training SVM with best choice of C and gamma')
    print('Shape of train data: {}'.format(Xtr.shape))
    print('Shape of train labels: {}'.format(Ytr.shape))
    print('Shape of test data: {}'.format(Xte.shape))
    print('Shape of test labels: {}'.format(Yte.shape))

    best_svm = SVC()
    best_c = 3
    best_g = 1000

    print('C: {}, gamma:{}'.format(best_c,best_g))

    best_svm.set_params(C=best_c,gamma=best_g)
    #best_svm.fit(Xtr,Ytr)
    Yte_pred = best_svm.predict(Xte)
    print('Test set accuracy of best svm: {}'.format(accuracy_score(Yte,Yte_pred)))
    print('Test set f1-score of best svm: {}'.format(f1_score(Yte,Yte_pred,average='macro')))
