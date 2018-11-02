import data_utils
import sys
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from copy import deepcopy

plt.switch_backend('agg')

# Load data
Xtr, Ytr, Xte, Yte = data_utils.load_cifar10('./cifar-10-batches-py')
num_train_samples = Xtr.shape[0]
num_test_samples = Xte.shape[0]

# Sanity checks
print('Shape of train data: {}'.format(Xtr.shape))
print('Shape of train labels: {}'.format(Ytr.shape))
print('Shape of test data: {}'.format(Xte.shape))
print('Shape of test labels: {}'.format(Yte.shape))

# Small development set for quick training and testing
num_dev_samples = 1000
mask = np.random.choice(num_train_samples, num_dev_samples, replace=False)
Xdev_tr = Xtr[mask]
Ydev_tr = Ytr[mask]

mask = np.random.choice(num_test_samples, int(num_dev_samples/2), replace=False)
Xdev_te = Xte[mask]
Ydev_te = Yte[mask]

# Preprocessing

# Unit normalization?
# Mean normalization
# PCA
# LDA
# Feature representations?

# SVM Classifier
svm = SVC()
best_svm = None
best_mean_accuracy = 0

# Hyperprameter search with cross-validation
c_choices = [10, 20] # TODO
g_choices = [0.02, 0.4] # TODO
num_folds = 5
Xtr_folds = np.split(Xdev_tr,num_folds)
Ytr_folds = np.split(Ydev_tr,num_folds)

cg_to_accuracies = {}

for c in c_choices:
    for g in g_choices:
        svm.set_params(C=c,gamma=g)
        cg_accuracy = np.zeros(num_folds)
        for i in range(num_folds):
            tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
            tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
            svm.fit(tr_data,tr_labels)
            cg_accuracy[i] = svm.score(Xtr_folds[i],Ytr_folds[i])

        if(np.mean(cg_accuracy) > best_mean_accuracy):
            best_mean_accuracy = np.mean(cg_accuracy)
            best_c = c
            best_g = g
            best_svm = deepcopy(svm)
        cg_to_accuracies[(c,g)] = cg_accuracy

for c,g in sorted(cg_to_accuracies):
    for accuracy in cg_to_accuracies[(c,g)]:
        print('c = {}, g = {}, accuracy = {}'.format(c,g,accuracy))

print('Best choice for C, gamma from cross-validation: {}, {}'.format(best_c,best_g))
print('Validation accuracies for best C, Gamma: {}'.format(cg_to_accuracies[(best_c,best_g)]))
print('Test set accuracy of best linear svm: {}'.format(best_svm.score(Xdev_te,Ydev_te)))
