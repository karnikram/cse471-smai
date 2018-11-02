import data_utils
import sys
import numpy as np
from sklearn.svm import LinearSVC
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
lsvm = LinearSVC()
best_lsvm = None
best_mean_accuracy = 0

# Hyperprameter search with cross-validation
c_choices = [5, 10] #TODO
num_folds = 5
Xtr_folds = np.split(Xdev_tr,num_folds)
Ytr_folds = np.split(Ydev_tr,num_folds)

c_to_accuracies = {}
for c in c_choices:
    lsvm.set_params(C=c)
    c_accuracy = np.zeros(num_folds)
    for i in range(num_folds):
        tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
        tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
        lsvm.fit(tr_data,tr_labels)
        c_accuracy[i] = lsvm.score(Xtr_folds[i],Ytr_folds[i])

    if(np.mean(c_accuracy) > best_mean_accuracy):
        best_mean_accuracy = np.mean(c_accuracy)
        best_c = c
        best_lsvm = deepcopy(lsvm)
    c_to_accuracies[c] = c_accuracy

for c in sorted(c_to_accuracies):
    for accuracy in c_to_accuracies[c]:
        print('c = {}, accuracy = {}'.format(c,accuracy))

for c in c_choices:
    accuracies = c_to_accuracies[c]
    plt.scatter([c]*len(accuracies),accuracies)

mean_accuracies = np.array([np.mean(a) for c,a in sorted(c_to_accuracies.items())])
std_accuracies = np.array([np.std(a) for c,a in sorted(c_to_accuracies.items())])
plt.errorbar(c_choices,mean_accuracies,yerr=std_accuracies)
plt.title('Cross-validation on C')
plt.xlabel('C')
plt.ylabel('Cross-validation accuracy')
plt.savefig('c-cross-validate.png',bbox_inches='tight')

print('Best choice for C from cross-validation: {}'.format(best_c))
print('Validation accuracies for best C: {}'.format(c_to_accuracies[best_c]))
print('Test set accuracy of best linear svm: {}'.format(best_lsvm.score(Xdev_te,Ydev_te)))
