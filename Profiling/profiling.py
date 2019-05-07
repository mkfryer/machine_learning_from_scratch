# profiling.py
"""
<Michael Fryer>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import timeit
import math
from numba import *
import time
from matplotlib import pyplot as plt

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        # read in data
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
        #reverse the data to backtrack through the triangle
        for i in reversed(range(len(data) - 1)):
            #replace each entry with the sum of the current entry 
            # and the greater of the two “child entries.” Continue 
            # this replacement up through the entire triangle.
            for j in range(len(data[i])):
                data[i][j] += max(data[i+1][j], data[i+1][j+1])
        #the top number contains the largest sum
        return data[0][0]

# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    #fill list with two to avoid computing it
    #since all primes are odd
    primes_list = [2]
    #number to test for division
    current = 3
    while len(primes_list) < N:
        isprime = True
        #skip even numbers and only check up to the square of the ceil number
        for i in range(3, int(math.sqrt(current)) + 1, 2):
            # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
                #if its not prime then there is no need to check again
                break
        if isprime:
            primes_list.append(current)
        #skip evens
        current += 2
    return primes_list

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #use array broadcasting to fin dthe min norm for each column
    return np.argmin(np.linalg.norm(A - x, axis=0))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    #load in data
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #total number of scores in file
    total = 0    
    for i, name in enumerate(names):
        #total number of scores in name
        name_value = 0
        for letter in name:
            #add value of each letter to word value
            name_value += alphabet.index(letter) + 1
        # add word totla to the over all score total
        total += (i + 1) * name_value
    return total


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    #initial fib numbers
    a = 1
    b = 1
    while True:
        #sreturn b in generation
        yield b
        # assign b to a and a to the next fib number
        a, b = b, a + b

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #cycle through generator returning fib 
    #numbers
    for n, fib in enumerate(fibonacci()):
        #checks if number of digits are less 
        #than N
        if int(math.log10(fib))+1 >= N:
            return n

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    a = range(2, N)
    while len(a) > 0:
        #assign first to the first of the list
        first = a[0]
        #generate first
        yield first
        #keep only values not divisible by a
        a = [x for x in a if x % first != 0]


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

#use compoilation to speed up python slugishness
@jit(nopython=True, locals=dict(A=double[:,:], product=double[:,:], m=int64, n=int64, temporary_array=double[:]))
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """

    x = 2 ** np.arange(2, 8)
    #list for matrix_power
    a = np.zeros(len(x))
    #list for matrix_power_numba
    b = a.copy()
    #list for linalg.matrix_power
    c = a.copy()

    #used to compile code for numba function
    matrix_power_numba(np.random.random((1,1)), 1)

    for i, m in enumerate(x):
        A = np.random.random((m,m))

        #time matrix_power
        start = time.time()
        matrix_power(A, n)
        a[i] = time.time() - start

        #time matrix_power_numba
        start = time.time()
        matrix_power_numba(A, n)
        b[i] = time.time() - start

        #time linalg matrix power
        start = time.time()
        np.linalg.matrix_power(A, n)
        c[i] = time.time() - start

    #graph three times side by side
    plt.loglog(x, a, label="matrix_power")
    plt.loglog(x, b, label="matrix_power_numba")
    plt.loglog(x, c, label="np.linalg.matrix_power")
    #show lables
    plt.legend()

    #plot axis labels
    plt.xlabel("Matrix size (mxm)")
    plt.ylabel("time (seconds)")

    #set title and show graph
    plt.title("Matrix Power Preformance")
    plt.show()












# A = np.random.rand(4,3)
# x = np.random.rand(4,1)

# print(A)
# print(x)
# print(A - x)
# print(x.shape)
# print(nearest_column(A,x))
# print(nearest_column_fast(A,x))




# print(name_scores())
# print(name_scores_fast())
