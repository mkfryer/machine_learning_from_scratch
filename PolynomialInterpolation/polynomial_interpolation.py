import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, barycentric_interpolate
from scipy import linalg as la
from numpy.fft import fft

def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    l = len(xint)
    r = range(l)

    def p(x):
        y = np.zeros(l)
        for j in r:
            #mask prohibits the product when xk = xj 
            mask = np.delete(np.arange(0, l), j)
            #Compute the denominator of each Lj (as in Equation 9.2) .
            #Using the previous step, evaluate each Lj at all points in the compuational domain
            y[j] += yint[j] * (np.prod(x - xint[mask])/np.prod(xint[j] - xint[mask]))
        #sum the evaluation at y
        return np.sum(y)

    return np.vectorize(p)(points)


class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #set domain and range interpolation points
        self.xint = xint
        self.yint = yint
        #number of interpolation points
        self.n = len(xint)
        #Barycentric weights 
        self.weights = np.empty(self.n)
        # Ccapacity of the interval
        self.c = (np.max(xint) - np.min(xint)) / 4
        #Barycentric weights 
        for j in range(self.n):
            self.weights[j] = 1 / np.prod(xint[j] - xint[np.arange(self.n) != j])
        #prevent overflow during computation.
        self.weights / self.c     
        
        

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        
        p = np.empty(len(points))
        t = np.linspace(-1, 1, self.n)

        for i in range(len(points)):
            #get scaling factor
            z = points[i] - t
            #prohibits the product when xk = xj 
            j = np.flatnonzero(z == 0)
            if j.size == 0:
                #scale weights
                a = self.weights / z
                p[i] = np.sum(a * self.yint) / np.sum(a)
            #if xk == xj
            else:
                p[i] = self.yint[j]
        return p

    def add_weights(self, new_xint, new_yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        self.weights *= self.c

        for i, xi in enumerate(new_xint):
            
            #calculate new weight
            wi = 1 / np.prod(xi - self.xint)

            #account fo rnew weight in othe weights
            for j, xj in enumerate(self.xint):
                self.weights[j] = self.weights[j]/(self.xint[j] - xi)
            
            #set new domain and range interpolation points
            self.xint = np.append(self.xint, values=[xi])
            self.yint = np.append(self.yint, values=[new_yint[i]])
            #Barycentric weights 
            self.weights = np.append(self.weights, values=[wi])
            # Ccapacity of the interval
            self.c = (np.max(self.xint) - np.min(self.xint)) / 4
            #prevent overflow during computation.
            self.weights /= self.c
        
        self.n = len(self.xint)

def test_barcentric_against_runges_function():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #domain for graph
    d = np.linspace(-1, 1, 400)
    #iterations for testing B interp. point effect
    N = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8]
    #function to approximate
    f = np.vectorize(lambda x: 1/(1+25*x**2))
    #to chevy, extrema
    ch_ext = lambda a, b, n: .5*(a + b + (b - a)*np.cos((np.arange(0, n) * np.pi)/n))
    #evenly spaced error margins
    e1 = []
    #cheby extram interp error margins
    e2 = []
    for n in N:

        #points to interpolate
        pts = np.linspace(-1, 1, n)
        #interpolate
        poly = BarycentricInterpolator(pts)
        poly.set_yi(f(pts))
        #get errror in terms of infinity norm
        e1.append(la.norm(f(d)-poly(d), ord=np.inf))

        #points to interpolate
        pts = ch_ext(-1, 1, n+1)
        #interpolate
        poly = BarycentricInterpolator(pts)
        poly.set_yi(f(pts))
        #get errror in terms of infinity norm
        e2.append(la.norm(f(d)-poly(d), ord=np.inf))

    #graph everything n terms of log base 10
    plt.xscale('log', basex=10)
    plt.yscale('log', basey=10)
    plt.plot(N, e1, label="uniform points")
    plt.plot(N, e2, label="chev. extrema")
    #set graph labels
    plt.xlabel("# of interp. points") 
    plt.ylabel("Error") 
    plt.title("Uniform vs Cheby interp. error")
    plt.legend()
    plt.show()


def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #chevy extremizers
    chubby_extrema = np.cos((np.pi * np.arange(n*2)) / n)
    #funciton evaluated at chev. extremizers
    samples = f(chubby_extrema)
    #fft cooeficients
    coeffs = np.real(fft(samples))[:n+1] / n
    #turn fft coeefecinets into cheb. coefficients
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs



def estimate_function_for_air_data(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plots the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #dataset of air quality
    data = np.load('airdata.npy')

    #chevy extrizers
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    #scaled intervals to apx chevy extremizers
    a, b = 0, 366 - 1/24
    #scaled domain
    domain = np.linspace(0, b, 8784)
    #apx chevy extrema with regars to actual data (non continuous data....)
    points = fx(a, b, n)
    #mask for extrema aprx
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    #interpolating poylynomial
    poly = BarycentricInterpolator(domain[temp2])
    poly.set_yi(data[temp2])

    #plot the data
    #set up plot settings
    plt.subplot(211)
    plt.title("PM_2.5 hourly concentration data 2016")
    plt.xlabel("Hours from Jan 1")    
    plt.ylabel("PM_2.5 level")
    plt.plot(data)    

    #domain for graph
    # x = np.linspace(0, 8784, 8784 * 10)
    x = np.linspace(0, b, len(domain))
    #plot the data
    #set up plot settings
    plt.subplot(212)
    plt.xlabel("Days from Jan 1")
    plt.ylabel("PM_2.5 level")
    plt.title("Aproximating Polynomial")

    #show graph
    plt.plot(x, poly(x))
    plt.tight_layout()
    plt.show()

