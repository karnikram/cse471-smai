import data_utils
from sklearn.svm import LinearSVC

# Load data
Xtr, Ytr, Xte, Yte = data_utils.load_cifar10('./cifar-10-batches-py')

num_train_samples = Xtr.shape[0]
num_validation_samples = int(num_train_samples * 0.2)

mask = range(num_validation_samples)
Xvald = Xtr[mask]
Yvald = Ytr[mask]

mask = range(num_validation_samples, num_training_samples)
Xtr = Xtr[mask]
Ytr = Ytr[mask]

# Small developement set for quick training
num_dev_samples = 1000
mask = np.random.choice(Xtr.shape[0], 500, replace=False)
Xdev = Xtr[mask]
Ydev = Ytr[mask]

# Sanity checks
print('Shape of train data: {}'.format(Xtr.shape))
print('Shape of train labels: {}'.format(Ytr.shape))
print('Shape of validation data: {}'.format(Xvald.shape))
print('Shape of validation labels: {}'.format(Yvald.shape))
print('Shape of test data: {}'.format(Xte.shape))
print('Shape of test labels: {}'.format(Yte.shape))

# SVM Classifier
clf = LinearSVC()
clf.fit(Xtr, Ytr)

#print(clf.score(Xte, Yte))
