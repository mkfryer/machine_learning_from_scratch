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


    def adapt_input(self, x):
        x = np.array(x)
        #account threshold function by appending -1 to input
        if len(x) != len(self.weights):
            x = np.hstack([x, [self.bias]])
        return x


    def predict(self, x):
        """ 
        t (float): target output
        x (list): input vector
        """
        x = self.adapt_input(x)
        #input for threshold function
        net = x @ self.weights
        #gets output of implicit threshold function
        output = 1 if net > 0 else 0

        return output
    
    def learn(self, x, t):
        """ 
        t (float): target output
        x (list): input vector
        """
        #account threshold function by appending -1 to input
        x = self.adapt_input(x)
        output = self.predict(x)

        #if perceptron did not evaluate correctly, update weights
        if output != t:
            """ """
            delta_weights = self.learning_rate * (t - output) * x
            self.weights += delta_weights

    def train(self, data, epochs):
        for _ in range(epochs):
            for x in data:
                self.learn(x[:-1], float(x[-1]))
        



def test():
    # data, meta = arff.loadarff("linearlySeperable.arff")
    # data=data.tolist()
    # P = Perceptron(.1, 2)
    # P.train(data, 10)
    # d2 = np.linspace(-.2, .2, 2)
    # y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]
    # data = np.array(data, dtype=np.float)

    # plt.scatter(data[4:, 0], data[4:, 1])
    # plt.scatter(data[:4, 0], data[:4, 1])
    # plt.plot(d2, y2)
    # plt.grid(True)
    # plt.title("Linearly Seperable")
    # plt.show()


    data, meta = arff.loadarff("linearlyUnseperable.arff")
    data=data.tolist()
    P = Perceptron(.1, 2)
    P.train(data, 10)
    d2 = np.linspace(-.2, .2, 2)
    y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]
    data = np.array(data, dtype=np.float)

    plt.scatter(data[4:, 0], data[4:, 1])
    plt.scatter(data[:4, 0], data[:4, 1])
    plt.plot(d2, y2)
    plt.grid(True)
    plt.title("Linearly Unseperable")
    plt.show()

    # dataset, meta = arff.loadarff("iris_training.arff")  
    # dataset = dataset.tolist()
    # P = Perceptron(.1, 4)
    # P.train(dataset, 1)
    # print(dataset)
    # success = 0
    # total = len(dataset[0,:])
    # for x in dataset:
    #     if P.predict(x[:-1]) == x[-1]:
    #         success+=1
    # print(success/total)
    # P = Perceptron(.5, 2, 100)
    # x1 = np.array([.8, .3])
    # x2 = np.array([.4, .1])
    # t = np.array([1, 0])
    # P.learn(x1, t[0])
    # P.learn(x2, t[1])

test()