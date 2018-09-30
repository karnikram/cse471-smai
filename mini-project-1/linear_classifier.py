import sys
import numpy as np
from PIL import Image

# Run pca on X and return reduced subspace U
def pca(X):

    # Compute subspace
    COV = np.dot(X.T,X) # since d > n
    U,S,V = np.linalg.svd(COV)
    U = np.dot(X,U)

    # Normalize the eigen vectors
    for i,col in enumerate(U.T):
        U[:,i] = col / np.linalg.norm(col) 

    return U

# Returns hypothesis vector
def softmax(x,W):

    h = np.array([np.exp(np.dot(w.T,x)) for w in W.T])
    normalization = sum([np.exp(np.dot(w.T,x)) for w in W.T])
    h = h / normalization

    return h

# Computes softmax cost
def softmax_cost(X,Y,classes,W):
    num_imgs = len(Y)
    num_classes = len(classes)

    c = 0

    for i in range(num_imgs):
        normalization = sum([np.exp(np.dot(w.T,X[:,i])) for w in W.T])
        for j in range(num_classes):
            c -= 1 * int(Y[i] == classes[j]) * np.log(np.exp(np.dot(W[:,j].T,X[:,i]))/normalization)

    return c

# Computes and returns the jacobian for minimization
def jacobian(X,Y,classes,W):
    num_imgs = len(Y)
    num_classes = len(classes)
    J = np.empty((W.shape[0],num_classes))

    for i in range(num_classes):
        j = np.zeros((W.shape[0]))
        for n in range(num_imgs):

            normalization = sum([np.exp(np.dot(w.T,X[:,n])) for w in W.T])
            j += -1 * X[:,n] * (1 * int(Y[n] == classes[i]) - np.exp(np.dot(W[:,i].T,X[:,n]))/normalization)

        J[:,i] = j

    return J

# Trains weight vectors W to minimize softmax cost function
def grad_desc(X,Y,classes,W):
    
    num_classes = len(classes)

    max_num_iters = 100
    l_rate = 0.000000001
    threshold = 0.001

    iter_no = 0
    J = jacobian(X,Y,classes,W)
    cost = softmax_cost(X,Y,classes,W)
    prev_cost = 0

    while((iter_no < max_num_iters) and abs(prev_cost - cost) > threshold):
        for i in range(num_classes):
            W[:,i] -= l_rate * J[:,i]

        prev_cost = cost
        iter_no += 1
        J = jacobian(X,Y,classes,W)
        cost = softmax_cost(X,Y,classes,W)

    return W

if __name__ == '__main__':

    try:
        train_file_path = sys.argv[1]
        test_file_path = sys.argv[2]

    except IndexError:
        print('Correct usage: linear_classifier.py <path-to-train-file> <path-to-test-file>')
        sys.exit(1)

    ##################
    ###  Training  ###
    ##################

    # Create X matrix of size d x n, list Y for labels
    n = sum(1 for l in open(train_file_path,'r'))
    im_w, im_h = Image.open(open(train_file_path,'r').readline().strip().split()[0]).size
    X = np.empty((im_w*im_h,n))
    Y = []

    # Load images from path into X matrix, labels into Y
    with open(train_file_path,'r') as f:
        for i, l in enumerate(f):
            im_path, label = l.strip().split()
            X[:,i] = np.array(Image.open(im_path).convert('L')).flatten()
            Y.append(label)

    classes = list(set(Y)) 

    # Mean normalization
    x_mean = np.sum(X,axis=1) / X.shape[1]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - x_mean

    # Run PCA and obtain the reduced representation of X, Z
    U = pca(X)
    k = 32
    U_red = U[:,:k]
    Z = np.dot(U_red.T,X)

    # Train for weights
    W = np.random.randn(k,len(classes)) * 0.0001 # initial estimate
    W_hat = grad_desc(Z,Y,classes,W)

    #################
    ###  Testing  ###
    #################
    
    # Input test images and tranform to reduced representation
    num_images = sum(1 for l in open(test_file_path,'r'))
    X = np.empty((im_h*im_w, num_images)) # assuming test images and train images have same shape

    with open(test_file_path,'r') as f:
        for i, l in enumerate(f):
            X[:,i] = np.array(Image.open(l.strip()).convert('L')).flatten()

    # Mean normalization
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - x_mean

    Z = np.dot(U_red.T,X)

    # Softmax classification
    for z in Z.T:
        h = softmax(z,W_hat)
        print(classes[np.argmax(h)])
