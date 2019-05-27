"""
Neural Network
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class MultilayeredPerceptron:
    def __init__(self, input_len, node_count):
        self.W1 = np.random.rand(node_count, input_len)
        self.W2 = np.random.rand(1, node_count)
        self.B1 = np.random.rand(node_count, 1)
        self.B2, = np.random.rand(1)

    def present_weights(self, ):
        print("W2", self.W2, "\n")
        print("W1", self.W1, "\n")
        print("B2", self.B2, "\n")
        print("B1", self.B1, "\n")

    def train(self, x, y):
        Z2, Z3, A2, A3 = self.forwardpass(x)
        self.backprop(Z2, Z3, A2, A3, x, y)

    def forwardpass(self, x):
        x_hat = np.append(x, [1])
        W1_hat = np.hstack((self.W1, self.B1.T))
        Z2 = W1_hat @ x_hat
        A2 = sigmoid(Z2)
        A2_hat = np.append(A2, [1])
        W2_hat = np.append(self.W2, 1)
        Z3 = W2_hat @ A2_hat
        A3 = sigmoid(Z3)
        return Z2, Z3, A2, A3

    def backprop(self, Z2, Z3, A2, A3, x, y):
        Delta3 = -(y - A3) * sigmoid_derivative(Z3)
        Delta2 = (self.W2 * Delta3) * sigmoid_derivative(Z2)

        D_W2 = Delta3 * A2
        D_W1 = np.dot(Delta2.T, x)
        D_B2 = Delta3
        D_B1 = Delta2
        
        self.W2 -= D_W2 
        self.W1 -= D_W1
        self.B2 -= D_B2
        self.B1 -= D_B1


def test_1():
    nn = MultilayeredPerceptron(2, 3)
    nn.W1 = np.ones((2, 2))
    nn.W2 = np.ones((1, 2))
    nn.B1 = np.ones((1, 2))
    nn.B2 = 1
    x = np.zeros((1,2))
    y = 1
    nn.train(x, y)
    x = np.array([0, 1]).reshape(1,2)
    y = 0
    nn.train(x, y)

    W1 = np.array([[1.0, 0.99449552], [1.0, 0.99449552]])
    W2 = np.array([[0.95812557, 0.95812557]])
    B2 = 0.9534321554039679
    B1 = np.array([[0.99561354, 0.99561354]])

    assert np.allclose(W1, nn.W1)
    assert np.allclose(W2, nn.W2)
    assert np.allclose(B1, nn.B1)
    assert np.allclose(B2, nn.B2)

    print("test 1 passed")

if __name__ == "__main__":
    test_1()
    
