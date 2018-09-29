from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Returns reduced subspace from X 
def pca(X):

    # Compute lower dimensional subspace
    cov = np.dot(X.T,X) * 1/X.shape[1]
    U,S,V = np.linalg.svd(cov) # Since d > N we compute svd on X.T*X
    U = np.dot(X,U) # Convert eigen vectors to d x 1
    
    # Normalize the eigen vectors
    for i, col in enumerate(U.T):
        U[:,i] = col / np.linalg.norm(col)

    return U

if __name__ == "__main__":

    try:
        file_path = sys.argv[1]

    except IndexError:
        print("Correct usage: python pca.py <image-list>.txt")
        print("<image-list>.txt line format: <image_path> <optional-label>")
        sys.exit(1)

    # Load images from the specified list and load into matrix
    n = sum(1 for l in open(file_path,'r'))
    im_w, im_h = Image.open(open(file_path,'r').readline().strip().split()[0]).size
    X = np.empty((im_w*im_h, n))

    with open(file_path,'r') as f:
        for i, l in enumerate(f):
            X[:,i] = np.array(Image.open(l.strip().split()[0]).convert('L')).flatten()

    # Mean normalize the samples
    x_mean = np.sum(X,axis=1) / X.shape[1]
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - x_mean

    U = pca(X) # dim - d x d

    #U_red = U[:,:32] # Select first 32 components
    #Z = np.dot(U_red.T,X)

    #X_new = np.dot(U_red,Z) # Reconstruct images

    #x_mean = np.sum(X_new,axis=1) / X.shape[1] 
    #for i in range(X_new.shape[1]):
        #X_new[:,i] = X_new[:,i] + x_mean

    #Image.fromarray(X_new[:,0].reshape(im_h,im_w)).show()
    #Image.fromarray(X[:,0].reshape(im_h,im_w)).show()

    # Compute MSE for different k
    K = np.arange(100,200)
    mse = np.empty(K.shape)

    #for i, k in enumerate(K):
        #U_red = U[:,:k] # Select k components
        #Z = np.dot(U_red.T,X)
        #X_new = np.dot(U_red,Z)

        # Compute MSE loss
        #sum = 0
        #for j in range(X.shape[1]):
            #sum = sum + np.linalg.norm(X[:,j] - X_new[:,j])**2

        #mse[i] = sum/X.shape[1]

    # Plot MSE vs K
    #plt.figure()
    #plt.plot(K,mse)
    #plt.xlabel('k')
    #plt.ylabel('mse')
    #plt.title('MSE vs K')
    #plt.show()

   # Plot 1D, 2D, 3D representations
    Z1 = np.dot(U[:,0].T,X)
    Z2 = np.dot(U[:,:2].T,X)
    Z3 = np.dot(U[:,:3].T,X)

    plt.figure(1)
    plt.scatter(Z1,np.ones(Z1.shape[0]))

    plt.figure(2)
    plt.scatter(Z2[0,:],Z2[1,:])

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z3[0,:],Z3[1,:],Z3[2,:])

    plt.show()
