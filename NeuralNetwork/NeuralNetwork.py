"""
Neural Network
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, input_len, hidden_layer_size, features, LR = .1):
        # mean and standard deviation
        mu, sigma = 0, 0.5
        self.feat_len = len(features)
        self.LR = LR
        self.W1 = np.random.normal(mu, sigma, size = (hidden_layer_size, input_len)).astype(np.float64)
        self.W2 = np.random.normal(mu, sigma, size = (self.feat_len, hidden_layer_size)).astype(np.float64)
        self.B1 = np.random.normal(mu, sigma, size = (hidden_layer_size, 1)).astype(np.float64)
        self.B2 = np.random.normal(mu, sigma, size = (self.feat_len, 1)).astype(np.float64)

    def present_weights(self ):
        print("W2", self.W2, "\n")
        print("W1", self.W1, "\n")
        print("B2", self.B2, "\n")
        print("B1", self.B1, "\n")
    
    def predict(self, x):
        _, _, _, y_hat = self.forwardpass(x)
        # print(y_hat.flatten())
        # print(np.argmax(y_hat))
        # print("guess:", y_hat)
        # return np.argmax(y_hat)
        return y_hat

    def train_singleton(self, x, y):
        Z2, Z3, A2, A3 = self.forwardpass(x)
        self.backprop(Z2, Z3, A2, A3, x, y)

    def get_errors(self, data):
        m, n = data.get_features().data.shape
        mse = 0
        acc = 0
        for j in range(m):
            x = data.get_features().data[j, :].reshape(4,1).astype(np.float64)
            y_idx = int(data.get_labels().data[j, 0])
            y = np.zeros((self.feat_len, 1)).reshape(3,1).astype(np.float64)
            y[y_idx, 0] = 1
            y_hat = self.predict(x)
            if np.argmax(y) == np.argmax(y_hat):
                acc += 1
            mse += np.linalg.norm(y - y_hat)**2

        acc /= m
        mse /= m
        return acc, mse


    def train_set(self, train_set, test_set, validation_set, w = 5):
        m1, n1 = train_set.get_features().data.shape
        min_acc = np.inf
        all_mse_te = []
        all_mse_va = []
        all_acc_va = []
        
        while True:
            
            for i in range(w):
                train_set.shuffle()
                validation_set.shuffle()
                train_set.shuffle()

                #train
                for j in range(m1):
                    x = train_set.get_features().data[j, :].reshape(4,1).astype(np.float64)
                    y_idx = int(train_set.get_labels().data[j, 0])
                    y = np.zeros((self.feat_len, 1)).reshape(3,1).astype(np.float64)
                    y[y_idx, 0] = 1
                    self.train_singleton(x, y)

                acc_va, mse_va = self.get_errors(validation_set)
                acc_te, mse_te = self.get_errors(test_set)

                all_acc_va.append(acc_va)
                all_mse_va.append(mse_va)
                all_mse_te.append(mse_te)

            new_min_acc = max(all_acc_va[-w:])

            if min_acc > new_min_acc:
                break
            else:
                min_acc = new_min_acc

        return all_acc_va, all_mse_va, all_mse_te
                

    def forwardpass(self, x):
        x_hat = np.vstack((x, [[1]]))
        W1_hat = np.hstack((self.W1, self.B1))
        Z2 = W1_hat @ x_hat
        A2 = sigmoid(Z2)
        A2_hat = np.vstack((A2, [[1]]))
        W2_hat = np.hstack((self.W2, self.B2))
        Z3 = W2_hat @ A2_hat
        A3 = sigmoid(Z3)
        return Z2, Z3, A2, A3

    def backprop(self, Z2, Z3, A2, A3, x, y):
        Delta3 = -(y - A3) * sigmoid_derivative(Z3)
        Delta2 = (self.W2.T @ Delta3) * sigmoid_derivative(Z2)

        D_W2 = self.LR * (Delta3 @ A2.T)
        D_W1 = self.LR * (Delta2 @ x.T)
        D_B2 = self.LR * (Delta3)
        D_B1 = self.LR * (Delta2)
        
        self.W2 -= D_W2 
        self.W1 -= D_W1
        self.B2 -= D_B2
        self.B1 -= D_B1


def test_basics():
    nn = NeuralNetwork(2, 2, [1], LR = 1)
    nn.W1 = np.ones((2, 2))
    nn.W2 = np.ones((1, 2))
    nn.B1 = np.ones((2, 1))
    nn.B2 = np.array([[1.0]])
    x = np.zeros((2, 1))
    y = np.array([1])
    nn.train_singleton(x, y)
    x = np.array([0, 1]).reshape(2,1)
    y = np.array([0])
    nn.train_singleton(x, y)

    W1 = np.array([[1.0, 0.99449552], [1.0, 0.99449552]])
    W2 = np.array([[0.95812557, 0.95812557]])
    B2 = 0.9534321554039679
    B1 = np.array([[0.99561354, 0.99561354]])

    assert np.allclose(W1, nn.W1, atol=1e-3)
    assert np.allclose(W2, nn.W2, atol=1e-3)
    assert np.allclose(B1, nn.B1, atol=1e-3)
    assert np.allclose(B2, nn.B2, atol=1e-3)

    print("test_basics passed")

if __name__ == "__main__":
    test_basics()
    
