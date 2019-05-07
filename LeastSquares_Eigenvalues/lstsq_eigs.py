# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from scipy import linalg as la
import math

def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = np.linalg.qr(A, mode = "reduced")
    y = np.dot(Q.T, b)
    x = np.zeros(len(b))
    #Use back substitution to solve Rx = y for x.
    x = solve_triangular(R, y)

    return x

# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])   
# A = np.vstack([x, np.ones(len(x))]).T
# x = least_squares(A, y)
# print(y)
# print(np.dot(A, x))

def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    d = np.load('housing.npy')

    x = d[:,0]
    y = d[:,1]
    A = np.vstack([x, np.ones(len(x))]).T
    lsq_s = least_squares(A, y)

    plt.plot(x, y, 'o', label='Original housing data', markersize=10)
    plt.plot(x, lsq_s[0]*x + lsq_s[1] , 'r', label='Fitted line')
    plt.xlabel('Year')
    plt.ylabel('price index')
    plt.legend()
    plt.show()

# line_fit()
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """ 
    d = np.load('housing.npy')

    x = d[:,0]
    x_space = np.linspace(d[0,0], d[-1,0])
    y = d[:,1]

    A = np.vander(x, 3)
    lsq_s = least_squares(A, y)
    p = np.poly1d(lsq_s)

    z = {3, 6, 9, 12}

    ax1 = plt.subplot(2,2,1)
    ax1.set_title("Poly degree 3")
    ax1.set_ylabel('price index')
    ax1.plot(x, y, 'o', label='Original housing data', markersize=10)
    ax1.plot(x_space, p(x_space), 'r')

    A = np.vander(x, 6)
    lsq_s = least_squares(A, y)
    p = np.poly1d(lsq_s)

    ax2 = plt.subplot(2,2,2)
    ax2.set_title("Poly degree 6")
    ax2.plot(x, y, 'o', label='Original housing data', markersize=10)
    ax2.plot(x_space, p(x_space), 'r')

    A = np.vander(x, 9)
    lsq_s = least_squares(A, y)
    p = np.poly1d(lsq_s)

    ax3 = plt.subplot(2,2,3)
    ax3.set_title("Poly degree 9")
    ax3.set_xlabel('year')
    ax3.set_ylabel('price index')
    ax3.plot(x, y, 'o', label='Original housing data', markersize=10)
    ax3.plot(x_space, p(x_space), 'r')
    
    A = np.vander(x, 12)
    lsq_s = least_squares(A, y)
    p = np.poly1d(lsq_s)

    ax4 = plt.subplot(2,2,4)
    ax4.set_title("Poly degree 12")
    ax4.set_xlabel('year')
    ax4.set_ylabel('price index')
    ax4.plot(x, y, 'o', label='Original housing data', markersize=10)
    ax4.plot(x_space, p(x_space), 'r')
    
    plt.show()
    

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    ell = np.load('ellipse.npy')
    x = ell[:,0]
    y = ell[:,1]
    A = np.vstack([x**2, x])
    A = np.vstack([A, x*y])
    A = np.vstack([A, y])
    A = np.vstack([A, y**2]).T
    z = np.ones(len(x))
    a, b, c, d, e = least_squares(A, z)
    plot_ellipse(a, b, c ,d ,e)
    plt.show()


def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")
