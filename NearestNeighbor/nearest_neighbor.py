import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mode
from matplotlib import pyplot as plt

def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #transpose z into a column and subtract it from each column of X
    A = X - z[np.newaxis, :]
    #square entries
    A = A**2
    #sum columns
    col_sums = np.sum(A, axis=1)
    #finish inner product
    inner_products = np.sqrt(col_sums)
    #grab index of smallest inner products
    i = np.argmin(inner_products)

    return X[i, :], inner_products[i]

class KDTNode:
    """General Node for a KDTree

    Attributes:
        value (ndarray): a numpy.ndarray array of an arbitrary size
        left (KDTNode) : node that is considered to be a "left" child
        right (KDTNode) : node that is considered to be a "right" child
        pivot (int): pivot position for comparison of children
    """
    def __init__(self, x, k = -1):
        """        
        Parameters:
        k (int): expected size of value
        z ((k, ) ndarray): a k-dimensional target point.
        """
        #validate objects being added as value
        if type(x) is not np.ndarray:
             raise TypeError(str(x) + " is not an ndarray")

        #validate node size
        if k != -1 and len(x) != k:
             raise ValueError(str(x) + " is not the right size")

        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """

        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        #if the tree is empty then assign the new node to root
        #make k, the length of the root node
        if self.root is None:
            self.k = len(data)
            self.root = KDTNode(data, self.k)
            self.root.pivot = 0
            return

        def next_pivot(current_pivot):
            """Returns the next pivot position for 
            the next row of the tree
            """
            if current_pivot == self.k -1 :
                return 0
            else:
                return current_pivot+1

        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            #if arrays are equal, we have a duplicate
            if np.array_equal(data, current.value): 
                raise ValueError(str(data) + " is already in the tree")
            elif data[current.pivot] < current.value[current.pivot]:
                if current.left is None: #add node if child does not exist
                   current.left = KDTNode(data, self.k)
                   current.left.pivot = next_pivot(current.pivot) 
                else:
                    return _step(current.left)  # if child exists, recurse down left side
            else:
                if current.right is None:  #add node if child does not exist
                     current.right = KDTNode(data, self.k)
                     current.right.pivot = next_pivot(current.pivot)
                else:
                    return _step(current.right) # if child exists, recurse down right side
                    
        return _step(self.root)


    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        #x, the closest vector to z is first assigned to the root
        x = self.root

        def closest(a, b):
            """Finds and returns the closest vector in terms of eaclidean
            distance with regards to z
            """
            if np.linalg.norm(a.value-z) < np.linalg.norm(b.value-z):
                return a
            else:
                return b

        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            nonlocal x # use x in parent scope
            if current is None: # Base case: dead end.
                return  
            elif z[current.pivot] < current.value[current.pivot]:
                x = closest(x, current) # assign x to the closest distance between x and current
                _step(current.left) # Recursively search left.
                #check for clsoser distance on the oposite side of the tree
                if z[current.pivot] + np.linalg.norm(x.value-z) >= current.value[current.pivot]:
                    _step(current.right)
                return
            else:
                x = closest(x, current) # assign x to the closest distance between x and current
                _step(current.right) # Recursively search right.
                #check for clsoser distance on the oposite side of the tree
                if z[current.pivot] - np.linalg.norm(x.value-z) <= current.value[current.pivot]:
                    _step(current.left)
                return
        _step(x)

        return x.value, np.linalg.norm(x.value-z)


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


class KNeighborsClassifier:
    """
    Classification wrapper object that gets training data for a 
    KDTree, fits the data, and establishes a prediction API with 
    said tree. 

    Attributes:
    n_neigbors (Int): the number of neighbors to include
        in the vote (the k in k-nearest neighbors)
    tree (KDTree): tree from which predictions are made
    lables ((,k)) ndarray): the training labels
    """
   
    def __init__(self, n_neighbors):
        """
        Parameters:
        n_neigbors (Int): refer to attributes
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Loads a SciPy KDTree with the data in X. Saves the 
            tree and the labels as attributes

        Parameters:
            X ((m, k) ndarray): a m-k-dimensional training set
            y ((,k)) ndarray): 1-dimensional NumPy array with m entries (the training labels)
        """
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """Paccept a 1-dimensional NumPy array z with k entries. 
        Query the KDTree for the n_neighbors elements of X that are nearest 
        to z and return the most common label of those neighbors.

        Parameters:
        z ((k,) ndarray): a k-dimensional target point.
        """
        #get indexes of closest vectors
        indices = self.tree.query(z, k = self.n_neighbors)[1]
        #if k = 1, then KDTree sends back an integer in place
        #of an array.
        if isinstance(indices, np.intp):
            indices = [indices]

        probable_labels = [self.labels[i] for i in indices]
        #Returns the most probable label and if there is a tie
        #it gives a alphanumeric priority
        return mode(probable_labels)[0]

def test_project(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """

    data = np.load("mnist_subset.npz")
    
    X_train = data["X_train"].astype(np.float) # Training data
    y_train = data["y_train"] # Training labels
    X_test = data["X_test"].astype(np.float) # Test data
    y_test = data["y_test"] # Test labels
    classifier = KNeighborsClassifier(n_neighbors) 
    classifier.fit(X_train, y_train) #train model

    #count number of correct predictions over the 
    #set X_test
    accuracy = 0
    for i in range(len(X_test)):
        if classifier.predict(X_test[i]) == y_test[i]:
            accuracy += 1

    return accuracy/len(y_test)



# print(type(prob6(1)))

# print("Predicted digit:", classifier.predict(X_test[i]), "Actual digit:", y_test[i], "Accuracy:", accuracy/max(1,i))

# print("Predicted Handwritten Digit:", classifier.predict(X_test[4]), "---Actual Value:", y_test[4])
# plt.imshow(X_test[4].reshape((28,28)), cmap="gray")
# plt.title("\nImage of Handwritten Digit:\n Predicted digit: " + str(classifier.predict(X_test[4])) + "\nActual digit: " + str(y_test[4]))
# plt.show()

# z = np.array([1,1])
# A = np.array([[2,2], [2,3]])
# A = np.random.random((100,10))
# z = np.random.random(10)
# print(z)
# print(z[np.newaxis,:])
# print(exhaustive_search(A, z))

# not be broadcast together with shapes (100,10) (10,1) 
# Problem 2: Write a KDTNode class.
# a = np.array([1,2,3])
# print(np.transpose(a))
# print(a.reshape(-1,1))


#for n in range(2, 15):
    # data = np.random.random((n,n))
    # z = np.random.random(n)
    # K = KDT()
    # for i in range(n):
    #     K.insert(data[i, :])
    # tree = KDTree(data)
    # a,b = tree.query(z)
    # c,d = K.query(z)
    # e,f = exhaustive_search(data, z)
    # if np.array_equal(a, c) and b == d:
    #     print(n, "True")
    #     print("a,b", a,b)
    #     print("c,d", c,d)
    #     # print("e,f", e,f)
    # else:
    #     print(n, "False")
    #     print("a,b", a,b)
    #     print("c,d", c,d)
        # print("e,f", e,f)

# a = np.array([3,4,1])
# b = np.array([1,2,7])
# c = np.array([4,3,5])
# d = np.array([2,0,3])
# e = np.array([2,4,5])
# f = np.array([6,1,4])
# g = np.array([1,4,3])
# h = np.array([0,5,7])
# i = np.array([5,2,5])

# K = KDT()
# K.insert(a)
# K.insert(b)
# K.insert(c)
# K.insert(d)
# K.insert(e)
# K.insert(f)
# K.insert(g)
# K.insert(h)
# K.insert(i)

# print(K)
# a = np.array([5,5])
# b = np.array([3,2])
# c = np.array([8,4])
# d = np.array([2,6])
# e = np.array([7,7])

# z = np.array([3, 2.75])

# K.insert(a)
# K.insert(b)
# K.insert(c)
# K.insert(d)
# K.insert(e)
# print(K.query(z))
