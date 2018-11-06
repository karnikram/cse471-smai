import data_utils
import argparse
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

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
    lda = LDA(n_components=100) #TODO
    lda.fit(Xtr,Ytr)
    lda.transform(Xtr)
    lda.transform(Xte)

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

# Logistic regression classifier
lr = LogisticRegression()
best_lr = None
best_mean_accuracy = 0

# K-fold cross-validation for hyperparameter C
c_choices = [0.1, 1, 10, 100]
num_folds = 5
Xtr_folds = np.split(Xdev_tr,num_folds) # split Xtr into 5 equal sub-arrays along axis 0
Ytr_folds = np.split(Ydev_tr,num_folds)

for c in c_choices:
    lr.set_params(C=c)
    c_accuracy = np.zeros(num_folds)
    for i in range(num_folds):
        tr_data = np.concatenate(Xtr_folds[:i]+Xtr_folds[i+1:])
        tr_labels = np.concatenate(Ytr_folds[:i]+Ytr_folds[i+1:])
        lr.fit(tr_data,tr_labels)
        c_accuracy[i] = lr.score(Xtr_folds[i],Ytr_folds[i])

    if(np.mean(c_accuracy) >= best_mean_accuracy):
        best_mean_accuracy = np.mean(c_accuracy)
        best_c = c
        best_lr = deepcopy(lr)
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
plt.ylabel('Accuracy')
plt.savefig('plots/lr-c-validation.png',bbox_inches='tight')

print('Best choice for C from 5-fold cross-validation: {}'.format(best_c))
print('Test set accuracy score of best Logistic regression classifier: {}'.format(best_lr.score(Xte,Yte)))
