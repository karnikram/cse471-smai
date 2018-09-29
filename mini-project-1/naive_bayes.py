from PIL import Image
import numpy as np
import sys

# Run pca on X and return reduced subspace U
def pca(X):
    
    # Compute subspace
    COV = np.dot(X.T,X) * 1/X.shape[1] # since d > n
    U,S,V = np.linalg.svd(COV)
    U = np.dot(X,U)

    # Normalize the eigen vector
    for i, col in enumerate(U.T):
        U[:,i] = col / np.linalg.norm(col)

    return U

def gaussian_density(x,mu,var):

    p = 1/(((2*np.pi)**1/2) * (var ** 1/2))* \
            np.exp(-1/2 * ((x - mu) / (var ** 1/2))**2)

    return p

if __name__ == '__main__':

    try:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]

    except IndexError:
        print('Correct usage: python naive_bayes.py <path-to-train-file> <path-to-test-file>')
        sys.exit(1)

    ##################
    ###  Training  ###
    ##################

    # Create X matrix of size d x n, list Y for labels
    n = sum(1 for l in open(train_file_path,'r'))
    im_w, im_h = Image.open(open(train_file_path,'r').readline().strip().split()[0]).size
    X = np.empty((im_w * im_h,n))
    Y = []

    # Load images from path into X matrix, labels into Y
    with open(train_file_path,'r') as f:
        for i, l in enumerate(f):
            im_path, label = l.strip().split()
            X[:,i] = np.array(Image.open(im_path).convert('L')).flatten()
            Y.append(label)

    # Mean normalization
    x_mean = np.sum(X,axis=1) / X.shape[1]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - x_mean

    # Run PCA and obtain the reduced representation of X, Z
    U = pca(X)
    k = 32
    U_red = U[:,:k]
    Z = np.dot(U_red.T,X)

    # Identify classes, and divide by classes
    classes = list(set(Y))
    num_classes = len(classes)
    Z_div = [[] for i in range(num_classes)] # Z_div structure : [[label1 images] [label2 images] ...]

    for i, y in enumerate(Y):
        for j, c in enumerate(classes):
            if(y == c):
                Z_div[j].append(Z[:,i]) # using lists for efficiency
                break
    
    # Estimate individual class likelihoods and their parameters
    means = np.empty((k,num_classes))
    variances = np.empty((k,num_classes))
    
    for i in range(num_classes):
        Z = np.array(Z_div[i]).T
        means[:,i] = np.sum(Z,axis=1)/Z.shape[1]
        variances[:,i] = np.var(Z,axis=1)
    
    #################
    ###  Testing  ###
    #################

    # Input test images and transform to reduced representation
    num_images = sum(1 for l in open(test_file_path,'r'))
    X = np.empty((im_h*im_w, num_images)) # assuming test images and train images have same shape

    with open(test_file_path,'r') as f:
        for i, l in enumerate(f):
            X[:,i] = np.array(Image.open(l.strip()).convert('L')).flatten()

    # Mean normalization
    for i in range(X.shape[1]):
        X[:,i] = X[;,i] - x_mean

    Z = np.dot(U_red.T,X)

    # Bayesian classification
    for z in Z.T:
        likelihoods = np.ones((num_classes))
        for i in range(num_classes):
            for j in range(k):
                likelihoods[i] *= gaussian_density(z[j],means[j,i],variances[j,i])

        print(likelihoods)
        print(classes[np.argmax(likelihoods)])
