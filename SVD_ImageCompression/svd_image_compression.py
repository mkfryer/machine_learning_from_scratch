"""
Michael Fryer

"""
import numpy as np
import numpy.linalg as linalg
import scipy.linalg
import matplotlib.pyplot as plt
from imageio import imread


# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #Calculate the eigenvalues and eigenvectors of AH A
    eigenValues, eigenVectors = linalg.eig(A.conj().T @ A)

    #compute signular values
    singular_values = np.sqrt(eigenValues)

    #reverse order from greatest to least
    #of eigen values with cooresponding vectors
    idx = singular_values.argsort()[::-1]
    singular_values = singular_values[idx]
    # print(singular_values)
    eigenVectors = eigenVectors[:,idx]

    tol_mask = singular_values >= tol
    #remove singular values and corresponding
    #vectors 
    singular_values = singular_values[tol_mask]
    eigenVectors = eigenVectors[:, tol_mask]

    #construct U
    U = (A @ eigenVectors)/singular_values

    return U, singular_values, eigenVectors.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #compute domain of unit circle
    D = np.linspace(0, 2*np.pi, 200)
    #create points fo rthe unit circle
    S = np.stack((np.cos(D), np.sin(D)))
    E = np.array([[1, 0, 0], [0, 0, 1]])

    #compute the SVD for the matrix A
    U,s,Vh = linalg.svd(A)
    #get the cooresponding matrix rep of the singular values
    s = np.diag(s)

    #plot S and E
    plt.subplot(2, 2, 1)
    plt.plot(S[0,:],S[1,:])
    plt.plot(E[0,:], E[1,:])
    plt.title("S")

    #V HS and V HE
    plt.subplot(2, 2, 2)
    plt.plot((Vh @ S)[0,:], (Vh @ S)[1,:])
    plt.plot((Vh @ E)[0,:], (Vh @ E)[1,:])
    plt.title("V^HS")
        
    #ΣV HS and ΣV HE
    plt.subplot(2, 2, 3)
    plt.plot((s @ Vh @ S)[0,:], (s @ Vh @ S)[1,:])
    plt.plot((s @ Vh @ E)[0,:], (s @ Vh @ E)[1,:])
    plt.title("ΣV^HS")

    #UΣV HS and UΣV HE
    plt.subplot(2, 2, 4)
    plt.plot((U @ s @ Vh @ S)[0,:], (U @ s @ Vh @ S)[1,:])
    plt.plot((U @ s @ Vh @ E)[0,:], (U @ s @ Vh @ E)[1,:])
    plt.title("UΣV^HS")

    plt.axis("equal")
    plt.subplots_adjust(hspace=.5)
    plt.show()



def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if np.linalg.matrix_rank(A) < s:
        raise ValueError("s > rank(A)")

    #get SVD decomposition of A
    U,Ep,Vh = linalg.svd(A)

    #strip excess ccolumns from U
    U_hat = U[:,0:s]
    #strip excess singular values
    Ep_hat = np.diag(Ep[0:s])
    #strip excess rows from Vh
    Vh_hat = Vh[0:s, :]

    #get count of total entries needed for SVD storage
    entry_count = U_hat.size + Ep[0:s].size + Vh_hat.size

    return U_hat @ Ep_hat @ Vh_hat, entry_count


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #get SVD decomposition of A
    U,Sig,Vh = linalg.svd(A)

    #get smallest singualar value less than error thresh.
    mask = Sig < err

    #get the next signuarl values index
    s = int(np.where(Sig == Sig[mask][0])[0])
    
    return svd_approx(A, s)

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename)/255
    # print(image)
    compressed_image = image

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    #if image is greyscale
    if len(image.shape) == 2:
        #get compressed image
        compressed_image, count = svd_approx(image, s)
        #set up graph information
        axs[0].set_title('Compressed')
        axs[0].imshow(compressed_image, cmap="gray")
        axs[0].axis("off")
        axs[1].set_title("Original")
        axs[1].imshow(image, cmap="gray")
        axs[1].axis("off")
        plt.suptitle('Difference: ' + str(image.size - count))

    else:
         #get compressed image layers
        compressed_red_layer, count_r = svd_approx(image[:,:,0], s)
        compressed_blue_layer, count_b  = svd_approx(image[:,:,1], s)
        compressed_green_layer, count_g = svd_approx(image[:,:,2], s)
        #stack compresed image layers
        compressed_image = np.dstack((compressed_red_layer, compressed_blue_layer, compressed_green_layer))
        #set up graph information
        axs[0].set_title('compressed')
        axs[0].axis("off")
        axs[0].imshow(compressed_image)
        axs[1].set_title("Original")
        axs[1].imshow(image)
        axs[1].axis("off")
        plt.suptitle('Size Difference:' + str(image.size - count_r + count_g + count_b))
    
    plt.show()

compress_image("hubble.jpg", 2)




# print(np.allclose(U.T @ U, np.identity(5)))
# print(np.allclose(U @ np.diag(s) @ Vh, A))
# print(np.linalg.matrix_rank(A) == len(s))


# A = np.array([[3,1], [1,3]])
# visualize_svd(A)
# Problem 3

# A = np.random.random((5,4))
# print(svd_approx(A, 3))


# A = np.random.random((10,5))
# print(A)
# print(lowest_rank_approx(A, .5))

