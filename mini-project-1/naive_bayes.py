from PIL import Image
import numpy as np
import sys

# Run pca on X and return reduced subspace U
def pca(X):
    # Mean normalization
    x_mean = np.sum(X,axis=1) / X.shape[1]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - x_mean

    # Compute lower dimensional subspace
    COV = np.dot(X.T,X) * 1/X.shape[1]
    U,S,V = np.linalg.svd(COV)
    U = np.dot(X,U)

    # Normalize the eigen vector
    for i,col in enumerate(U.T):
        U[:,i] = col / np.linalg.norm(col)

    return U

def gaussian_density(X,MU,COV):
    d = X.shape[0]
    np.linalg.inv(COV)

    p = 1/(((2*np.pi)**d/2) * (np.linalg.det(COV))**1/2) * \
            np.exp(-1/2 * (X - MU).T * np.linalg.inv(COV) * (X - MU))

    return p

if __name__ == '__main__':

    try:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]

    except IndexError:
        print('Usage: naive_bayes.py <path-to-train-file> <path-to-test-file>')
        sys.exit(1)

    ##################
    ###  Training  ###
    ##################

    # Create X matrix of size d x n, list Y for labels
    n = sum(1 for l in open(train_file_path,'r'))
    im_w, im_h = Image.open(open(train_file_path,'r').readline().split(' ')[0]).size
    X = np.empty((im_w * im_h,n))
    Y = []

    # Load images from path into X matrix, labels into Y
    with open(train_file_path,'r') as f:
        for i, l in enumerate(f):
            ls = l.split(' ')
            X[:,i] = np.array(Image.open(ls[0]).convert('L')).flatten()
            Y.append(ls[1].split('\n')[0])

    # Run PCA and obtain the reduced representation of X, Z
    U = pca(X)
    U_red = U[:,:32]
    Z = np.dot(U_red.T,X)

    # Identify classes, and divide by classes
    labels = list(set(Y))
    Z_div = [[] for i in range(len(labels))] # Z_div structure : [[label1 images] [label2 images] ...]

    for i, y in enumerate(Y):
        for j, l in enumerate(labels):
            if(y == l):
                Z_div[j].append(Z[:,i])
                break

    # Estimate parameters of individual densities

    # Bayesian classification

    ##################
    #### Testing #####
    ##################

