# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Michael Fryer>
<Class>
<Date>
"""
import numpy as np
from scipy import linalg as la
from scipy import stats
import matplotlib.pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #get normal distribution
    points = np.random.uniform(-1, 1, (n,N))
    #get list of indicies inside n-ball
    mask = la.norm(points, axis=0) <= 1
    #count list of indicies inside n-ball
    inner_count = np.count_nonzero(mask)
    #estimate volume of n-ball
    return 2**n * (inner_count / N) 


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #sample randomlly over interval from a to b
    d = np.random.uniform(a, b, N)
    #constant to get average
    c =  ((b - a) * (1 / N))
    #estimate integral using the average
    return c * sum(f(d))


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #cast mins, maxs to numpy arrays
    mins = np.array(mins)
    maxs = np.array(maxs)

    #get normal distribution
    points = np.random.uniform(0, 1, (N, len(mins)))
    bound_length = maxs - mins

    #Volume of multi-dim cube
    v = np.prod(maxs - mins)

    #scale points to new bounds
    shifted_points = (points * bound_length) + mins
    
    #evaluate function at each shifted point
    Y = [f(x) for x in shifted_points]

    #calc estimated integral proportioanlly to cube
    return (v / N) * sum(Y)

# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #dim of current space
    n = 4

    #bounds for integration
    min_bounds = [-3/2, 0, 0, 0]
    max_bounds = [3/4, 1, 1/2, 1]

    #PDF function
    pdf = lambda x: (1/((2*np.pi)**(n/2))) * np.exp(-(x @ x)/2)

    #get the true value of the integragtion
    means, cov = np.zeros(n), np.eye(n)
    t = stats.mvn.mvnun(min_bounds, max_bounds, means, cov)

    #compute estimates of the integral
    N = np.logspace(1, 5, num=20, base=10, dtype=np.int64)
    estimates = np.array([mc_integrate(pdf, min_bounds, max_bounds, log_step) for log_step in N])

    #compute relative error in estimates
    relative_error = np.abs(estimates - t[0])

    #comparitor for reference
    comparitor = 1/np.sqrt(N)

    #plot error as cont. function
    plt.loglog(N, relative_error, label='Relative Error')
    plt.loglog(N, comparitor, label='1/sqrt(N)')

    #plot error as scatter
    plt.scatter(N, relative_error)
    plt.scatter(N, comparitor)

    #graph settings
    plt.title("Monte Carlo PDF Error")
    plt.legend()
    plt.show()





















################
####tests#######
################

#1
# print(ball_volume(4, N=100000))

#2
# f = lambda x: x**2
# print(mc_integrate1d(f, -4, 2))

#3
# f = lambda x: 3*x[0] - 4*x[1] + x[1]**2
# mins = [1, -2]
# maxs = [3, 1]
# print(mc_integrate(f, mins, maxs, N=10000))

#4
# prob4()
