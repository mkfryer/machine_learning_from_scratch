# drazin.py
"""Volume 1: The Drazin Inverse.
<Michael Fryer>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_power
from scipy.sparse import csgraph
import csv

# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """

    #commutativity
    c1 = np.allclose(A @ Ad, Ad @ A)
    #other basic drazin properties....
    c2 = np.allclose(matrix_power(A, k+1) @ Ad, matrix_power(A, k))
    c3 = np.allclose(Ad @ A @ Ad, Ad)

    return c1 and c2 and c3


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    #shape of A
    m, n = np.shape(A)

    #sorting methods to sort columns of schur decomp
    g_sort = lambda x: abs(x) > tol
    l_sort = lambda x: abs(x) <= tol

    #schur decompositions
    Q1, S, k1 = la.schur(A, sort=g_sort)
    Q2, T, k2 = la.schur(A, sort=l_sort)

    #concatenate part of S and T column wise
    U = np.hstack((S[:,:k1], T[:,:n-k1]))
    U_inv = la.inv(U)
    V = U_inv @ A @ U
    Z = np.zeros((n, n))

    #compute drazin inverse
    if k1 != 0:
        M_inv = la.inv(V[:k1, :k1])
        Z[:k1, :k1] = M_inv
    
    return U @ Z @ U_inv

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    #shape of A
    n = A.shape[0]
    #Laplacian of A
    L = csgraph.laplacian(A, normed=False)
    #Resistance matrix to fill up
    R = np.zeros((n, n))
    I = np.eye(n)

    #iterate through R
    for i in range(n):
        for j in range(n):
            #diagonal should be zero
            if i != j:
                Lj = L.copy()
                #replace jth row in Lj with Jth row in ident.
                Lj[j, :] = I[j, :]
                #compute drazin inverse of new Lj
                Ljd = drazin_inverse(Lj)
                #replace entry
                R[i, j] = Ljd[i, i]

    return R



# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        with open(filename, 'r') as csvfile:
            rows = np.array(list(csv.reader(csvfile)))

            #flatten all name tuples, cast to set to 
            #get unique names and cast to list for ordering
            self.names = list(set(rows.flatten()))

            #network adjacency matrix
            n = len(self.names)
            self.A = np.zeros((n , n))

            #populate network
            for c in rows:
                #get cooresponding indicies of names in name list
                i = self.get_name_index(c[0])
                j = self.get_name_index(c[1])
                #make each bi-directional connection
                self.A[i, j] = 1
                self.A[j, i] = 1

            #resistance of the network A
            self.R = effective_resistance(self.A)
            
    def get_name_index(self, v):
        """
            Return matrix index of associated name
        """
        return self.names.index(v)

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        Rc = self.R.copy()

        #get all indicies of node connections
        connected_nodes_mask = self.A == 1

        #igonore connected nodes
        Rc[connected_nodes_mask] = 0

        def get_min_non_zero(Rc):
            """
            return value of smallest nonzero entry in Rc
            """
            return np.min(Rc[Rc > 0])


        if node is None:
            #get smallest value
            min_v = get_min_non_zero(Rc)

            #get cooresponding index of smallest value
            i, j = np.where(Rc == min_v)

            return self.names[i[0]], self.names[j[0]]

        else:
            j = self.get_name_index(node)
            #jth column of Rc
            Rcj = Rc[:, j]

            #get value of smallest nonzero entry in Rcj
            min_v = get_min_non_zero(Rcj)

            #get cooresponding index of smallest value
            i, = np.where(Rcj == min_v)

            return self.names[i[0]]



    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        #get indicies of names on the matrix
        i = self.get_name_index(node1)
        j = self.get_name_index(node2)

        #set bi-directional connection
        self.A[i, j] = 1
        self.A[j, i] = 1
            
        #resistance of the network A
        self.R = effective_resistance(self.A)


def test():
    #prob1
    A  = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    Ad = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])
    B = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
    Bd = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    Cd = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert (is_drazin(A, Ad, 1) == True)
    assert (is_drazin(B, Bd, 3) == True)
    assert (is_drazin(B, Cd, 3) == False)
    print("prob1 success \n")

    #prob2
    A  = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    Ad = drazin_inverse(A)
    assert (is_drazin(A, Ad, 1) == True)
    # A = np.array([[ 10,  -8,   6,  -3], [ 12, -10,   8,  -4], [  1,  -1,   1,   0,], [ -2,   2,  -2,   2]], dtype=float)
    A = np.array([[ 5, -3,  2], [15, -9,  6,], [10, -6,  4,]])
    Ad = drazin_inverse(A)
    # print(Ad)
    print("prob2 success \n")

    #prob3
    G1 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    R_test1 = effective_resistance(G1)
    R_true1 = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    G2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    R_test2 = effective_resistance(G2)
    assert (np.allclose(R_test1, R_true1))
    assert (np.allclose(R_test2[0, 1], 2/3))
    print("prob3 success \n")

    #prob 4/5

    #test predict link
    LP = LinkPredictor()
    nodes = LP.predict_link()
    assert("Emily" in nodes and "Oliver")
    assert LP.predict_link("Melanie") == "Carol"
    
    #test add_link
    LP.A[1, 2] = 0
    LP.A[2, 1] = 0
    n1 = LP.names[1]
    n2 = LP.names[2]
    LP.add_link(n1 , n2)
    assert LP.A[1, 2] == LP.A[2, 1] == 1
    print("prob4/5 success \n")



# test()