from scipy.io import arff
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Basic perceptron"""

    # learning_rule = lambda c, t, z, x: c * (t - z) * x

    def __init__(self, learning_rate, input_node_count, bias = 1):
        self.learning_rate = learning_rate
        self.input_node_count = input_node_count
        #initialize weights as zeros and increase dimensionality
        #by adding bias as node to simplify learning
        self.weights = np.zeros(input_node_count + 1)
        self.bias = bias
        # self.weights[-1] = -bias
        # print(self.weights)

    def predict(self, x):
        #account threshold function by appending -1 to input
        if len(x) != len(self.weights):
            x = np.hstack([x, [self.bias]])
        #input for threshold function
        net = x @ self.weights
        #gets output of implicit threshold function
        output = 1 if net > 0 else 0

        return output
    
    def learn(self, x, t):
        """ """
        #account threshold function by appending -1 to input
        x = np.hstack([x, [self.bias]])
        output = self.predict(x)

        #if perceptron did not evaluate correctly, update weights
        if output != t:
            """ """
            delta_weights = self.learning_rate * (t - output) * x
            # print("pattern", x, "delta weight", delta_weights)
            self.weights += delta_weights

        
def test():
    data, meta = arff.loadarff("linearlySeperable.arff")  
    P = Perceptron(.1, 2)
    for x1, x2, t in data:
        x = np.array([x1, x2])
        P.learn(x, int(t))

    d = [x[0] for x in data]
    y = [x[1] for x in data]
    d2 = np.linspace(-2, 2, 8)
    y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]
    print(P.weights)
    plt.scatter(d, y)
    plt.plot(d2, y2)
    plt.show()

    P = Perceptron(.5, 2, 100)
    # x1 = np.array([.8, .3])
    # x2 = np.array([.4, .1])
    # t = np.array([1, 0])
    # P.learn(x1, t[0])
    # P.learn(x2, t[1])

test()