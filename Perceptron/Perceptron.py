from scipy.io import arff
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as la
from scipy.io import arff

class Perceptron:
    """Basic perceptron"""

    def __init__(self, learning_rate, input_node_count, bias = 1.0):
        self.learning_rate = learning_rate
        self.input_node_count = input_node_count
        #initialize weights as zeros and increase dimensionality
        #by adding bias as node to simplify learning
        self.weights = np.zeros(input_node_count + 1)
        self.bias = bias

    def adapt_input(self, x):
        x = np.array(x, dtype=np.float64)
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
        return output, net
    
    def learn(self, x, t):
        """ 
        t (float): target output
        x (list): input vector
        """
        #account threshold function by appending -1 to input
        x = self.adapt_input(x)
        output, net = self.predict(x)

        #if perceptron did not evaluate correctly, update weights
        if output != t:
            """ """
            delta_weights = self.learning_rate * (t - output) * x
            self.weights += delta_weights

    def train(self, tr_data, te_data, tol=13E-3, max_epocs = 25):
        epocs = -1
        errors = []
        # accuracy = []
        err_variance_prev = 0
        errors.append(self.test(te_data))

        while True:
            epocs += 1
            for x in tr_data:
                self.learn(x[:-1], float(x[-1]))

            errors.append(self.test(te_data))
            m = sum(errors)/len(errors)
            err_variance = la.norm(np.array(errors) - m, ord=2) if len(errors) > 2 else np.inf
            
            if abs(err_variance - err_variance_prev) <= tol or epocs >= max_epocs:
                break
            err_variance_prev = err_variance
 
        return 1 - np.array(errors), epocs

    def test(self, data):
        success = 0
        for x in data:
            prediction, net = self.predict(x[:-1])
            if prediction == x[-1]:
                success += 1
        
        err = 1.0 - success/len(data[:,0])
        return err

# a = np.array(['handicapped-infants'
# , 'water-project-cost-sharing', 
# 'adoption-of-the-budget-resolution',
# 'physician-fee-freeze',
# 'el-salvador-aid',
# 'religious-groups-in-schools',
# 'anti-satellite-test-ban'
# 'aid-to-nicaraguan-contras'
# 'mx-missile'
# 'immigration'
# 'synfuels-corporation-cutback'
# 'education-spending'
# 'superfund-right-to-sue'
# 'crime'
# 'duty-free-exports'
# 'export-administration-act-south-africa'
# ])

"""
** -3.00000000e-01   @attribute 'handicapped-infants' { 'n', 'y'}
0.00000000e+00  @attribute 'water-project-cost-sharing' { 'n', 'y'}
**  5.00000000e-01  @attribute 'adoption-of-the-budget-resolution' { 'n', 'y'}
-1.30000000e+00  @attribute 'physician-fee-freeze' { 'n', 'y'}
-2.00000000e-01  @attribute 'el-salvador-aid' { 'n', 'y'}
 1.00000000e-01 @attribute 'religious-groups-in-schools' { 'n', 'y'}
 -2.77555756e-17  @attribute 'anti-satellite-test-ban' { 'n', 'y'}
** -5.00000000e-01 @attribute 'aid-to-nicaraguan-contras' { 'n', 'y'}
 1.00000000e-01  @attribute 'mx-missile' { 'n', 'y'}
-1.00000000e-01  @attribute 'immigration' { 'n', 'y'}
**  9.00000000e-01  @attribute 'synfuels-corporation-cutback' { 'n', 'y'}
** -3.00000000e-01 @attribute 'education-spending' { 'n', 'y'}
  2.77555756e-17  @attribute 'superfund-right-to-sue' { 'n', 'y'}
-2.00000000e-01  @attribute 'crime' { 'n', 'y'}
**  5.00000000e-01 @attribute 'duty-free-exports' { 'n', 'y'}
  2.77555756e-17 @attribute 'export-administration-act-south-africa' { 'n', 'y'}
**   3.00000000e-01 @attribute 'Class' { 'democrat', 'republican'}
"""

# part5()

# def plot(data, weights, title):
#     d2 = np.linspace(-1, 1, 2)
#     y2 = [(-weights[2] - x*weights[0])/weights[1] for x in d2]

#     plt.scatter(data[4:, 0], data[4:, 1])
#     plt.scatter(data[:4, 0], data[:4, 1])
#     plt.plot(d2, y2)
#     plt.grid(True)
#     plt.title(title)
#     plt.show()
# part3()

# def test():
#     data, meta = arff.loadarff("linearlySeperable.arff")
#     data = np.array(data.tolist(), dtype=np.float)
#     P = Perceptron(.1, 2)
#     P.train(data, 10)
#     plot(data, P.weights, "linearly Seperable")

#     data, meta = arff.loadarff("linearlyUnseperable.arff")
#     data = np.array(data.tolist(), dtype=np.float)
#     P = Perceptron(.1, 2)
#     P.train(data, 10)
#     plot(data, P.weights, "linearly Seperable")

# test()