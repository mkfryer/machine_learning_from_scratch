import numpy as np
import scipy
import scipy.sparse
from imageio import imread
import matplotlib.pyplot as plt
from scipy.sparse import linalg as spla

def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    return np.diag(A.sum(axis = 0)) - A



def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    eig_values = scipy.linalg.eig(laplacian(A))
    connected_comp_number = sum(1 for i in np.eig_values if np.real(i) < tol)
    algebraic_connectivity = np.max(np.partition(eig_values, 1))
    return connected_comp_number, algebraic_connectivity
    


def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


class ImageSegmenter:
    """Class for storing and segmenting images."""

    def __init__(self, filename = "dream.png"):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename)
        self.image = self.image / 255
        if len(self.image.shape) == 3:
            self.brightness = np.ravel(self.image.mean(axis=2))
        else:
            self.brightness = np.ravel(self.image)

    def show_original(self):
        """Display the original image."""
        if len(self.image.shape) == 3:
            plt.show(self.image, cmap="gray")
        else:
            plt.show(self.image)

    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        if len(self.image.shape) == 3:
            m, n, z = self.image.shape
        else:
            m, n = self.image.shape        
        A = scipy.sparse.lil_matrix((n*m, n*m))
        print(A)
        D = np.zeros(m*n)
        for i in range(m*n):
            neigbors, distances = get_neighbors(i, r, m, n)
            weights = np.exp(-1 * abs(self.brightness[i] - self.brightness[neigbors])/sigma_B2 - distances/sigma_X2)
            A[i, neigbors] = weights
            D[i] = sum(weights)
        
        return A.tocsc(), D

    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        if len(self.image.shape) == 3:
            m, n, z = self.image.shape
        else:
            m, n = self.image.shape
        L = scipy.sparse.csgraph.laplacian(A)
        D_1_2 = scipy.sparse.diags(D**-.5)
        DLD = D_1_2 @ L @ D_1_2
        print(DLD)
        vals, vecs = spla.eigsh(DLD, which="SM",  k=2)
        print((m,n))
        mask = vecs[:,1].reshape((m,n)) > 0
        return mask

    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)
        if len(self.image.shape) == 3:
            pos = self.image * np.dstack((mask, mask, mask))
            neg = self.image * np.dstack((~mask, ~mask, ~mask))
            plt.subplot(1,3,1)
            plt.imshow(self.image)
            plt.subplot(1,3,2)
            plt.imshow(pos)
            plt.subplot(1,3,3)
            plt.imshow(neg)
            plt.show()
        else:
            plt.subplot(1,3,1)
            plt.imshow(self.image, cmap="gray")
            plt.subplot(1,3,2)
            plt.imshow(self.image * mask, cmap="gray")
            plt.subplot(1,3,3)
            plt.imshow(self.image * ~mask, cmap="gray")
            plt.show()

a = ImageSegmenter()
a.segment()

# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
