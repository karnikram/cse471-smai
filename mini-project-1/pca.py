from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

# Returns reduced subspace from X 
def pca(X):

    # Mean normalize the samples
    x_mean = np.sum(X,axis=0) / X.shape[1]
    X = X - x_mean

    # Compute lower dimensional subspace
    cov = np.dot(X.T,X) * 1/X.shape[1]
    U,s,Vh = np.linalg.svd(cov) # Since d > N we compute svd on X.T*X
    U = np.dot(X,U) # Convert eigen vectors to d x 1
    
    print(U.shape)

    # Normalize the eigen vectors
    for i,col in enumerate(U.T):
        col = col / np.linalg.norm(col)
        U[:,i] = col

    return U

if __name__ == "__main__":

    try:
        file_path = sys.argv[1]

    except IndexError:
        print("Usage: pca.py <image-list>.txt")
        print("<image-list>.txt line format: <image_path> <label>")
        sys.exit(1)

    # Down sample for memory reasons 
    scale = 0.5
    im_w, im_h = Image.open(open(file_path,'r').readline().split(' ')[0]).size
    s_w = int(im_w * scale)
    s_h = int(im_h * scale)

    # Load images from the specified list and load into matrix
    n = sum(1 for l in open(file_path,'r'))
    X = np.empty((s_w*s_h, n))

    with open(file_path,'r') as f:
        for i, l in enumerate(f):
            ls = l.split(' ')
            im = Image.open(ls[0]).convert('L').resize((s_w,s_h))
            X[:,i] = np.array(im).flatten()

    U = pca(X) # dim - d x d
    
    U_red = U[:,:32] # Select first 32 components
    Z = np.dot(U_red.T,X)
    X_new = np.dot(U_red,Z) # Reconstruct images
    Image.fromarray(X_new[:,5].reshape(s_h,s_w)).show()


    # Compute MSE for different k
    #K = np.arange(10,400)
    #MSE = np.empty(K.shape)

    #for i, k in enumerate(K):
        #U_red = U[:,1:k] # Select k components
        #Z = np.dot(U_red.T,X)
        #X_new = np.dot(U_red,Z)

        # Compute MSE loss
        #sum = 0
        #for j in range(X.shape[1]):
            #sum = sum + np.linalg.norm(X[:,j] - X_new[:,j])**2

        #MSE[i] = sum/X.shape[1]

    # Plot MSE vs K
    #plt.figure()
    #plt.plot(K,MSE)
    #plt.xlabel('k')
    #plt.ylabel('mse')
    #plt.title('MSE vs K')
    #plt.show()
