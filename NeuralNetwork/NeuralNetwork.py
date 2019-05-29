"""
Neural Network
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def matrix_mult(A, b):
        if type(b) != np.ndarray or type(A) != np.ndarray:
            return A * b
        # elif type(b) == np.ndarray and b.shape == (1, 1) :
        #     return A * b[0][0]
        else:
            return A @ b


class NeuralNetwork:
    def __init__(self, input_len, hidden_layer_dims, feat_len, LR = .1, momentum = 0):
        # mean and standard deviation
        mu, sigma = 0, 0.5
        hidden_layer_dims.insert(0, input_len)
        hidden_layer_dims.append(feat_len)
        self.momentum = momentum
        self.feat_len = feat_len
        self.LR = LR
        self.input_len = input_len
        self.weights_prev = []
        self.weights = []
        self.biases = []
        self.biases_prev = []
        for i in range(1, len(hidden_layer_dims)):
            m = hidden_layer_dims[i]
            n = hidden_layer_dims[i-1]
            self.biases.append(
                np.random.normal(mu, sigma, size = (m, 1)).astype(np.float64)
            )
            self.biases_prev.append(
                np.zeros((m, 1))
            )
            self.weights.append( 
                np.random.normal(mu, sigma, size = (m, n)).astype(np.float64)
            )
            self.weights_prev.append(
                np.zeros((m, n))
            )
        # self.biases.append(np.random.normal(mu, sigma, size = (self.feat_len, 1)).astype(np.float64))
        # self.biases_prev.append(np.zeros((self.feat_len, 1)))
        # self.weights.append(np.random.normal(mu, sigma, size = (self.feat_len, hidden_layer_dims[-1])).astype(np.float64))
        # self.weights_prev.append(np.zeros((self.feat_len, hidden_layer_dims[-1])))

        # self.W1_prev = np.zeros((hidden_layer_size, input_len))
        # self.W2_prev = np.zeros((self.feat_len, hidden_layer_size))
        # self.B2_prev = np.zeros((self.feat_len, 1))
        # self.B1_prev = np.zeros((hidden_layer_size, 1))
        # self.W1 = np.random.normal(mu, sigma, size = (hidden_layer_size, input_len)).astype(np.float64)
        # self.W2 = np.random.normal(mu, sigma, size = (self.feat_len, hidden_layer_size)).astype(np.float64)
        # self.B1 = np.random.normal(mu, sigma, size = (hidden_layer_size, 1)).astype(np.float64)
        # self.B2 = np.random.normal(mu, sigma, size = (self.feat_len, 1)).astype(np.float64)

    # def present_weights(self ):
    #     print("W2", self.W2, "\n")
    #     print("W1", self.W1, "\n")
    #     print("B2", self.B2, "\n")
    #     print("B1", self.B1, "\n")
    
    def predict(self, x):
        _, A = self.forwardpass(x)
        # print(y_hat.flatten())
        # print(np.argmax(y_hat))
        # print("guess:", y_hat)
        # return np.argmax(y_hat)
        return A[-1]

    def train_singleton(self, x, y):
        # Z2, Z3, A2, A3 = self.forwardpass(x)
        # self.backprop(Z2, Z3, A2, A3, x, y)
        Z, A = self.forwardpass(x)
        self.backprop(Z, A, x, y)

    def get_errors(self, data):
        m, n = data.get_features().data.shape
        mse = 0
        acc = 0
        for j in range(m):
            x = data.get_features().data[j, :].reshape(self.input_len ,1).astype(np.float64)
            y_idx = int(data.get_labels().data[j, 0])
            y = np.zeros((self.feat_len, 1)).reshape(self.feat_len, 1).astype(np.float64)
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
        min_mse = np.inf
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
                    x = train_set.get_features().data[j, :].reshape(self.input_len,1).astype(np.float64)
                    y_idx = int(train_set.get_labels().data[j, 0])
                    y = np.zeros((self.feat_len, 1)).reshape(self.feat_len,1).astype(np.float64)
                    y[y_idx, 0] = 1
                    self.train_singleton(x, y)

                acc_va, mse_va = self.get_errors(validation_set)
                acc_te, mse_te = self.get_errors(test_set)

                all_acc_va.append(acc_va)
                all_mse_va.append(mse_va)
                all_mse_te.append(mse_te)

            new_min_mse = min(all_mse_va[-w:])

            if min_mse < new_min_mse:
                break
            else:
                min_mse = new_min_mse

        return all_acc_va, all_mse_va, all_mse_te
                

    def forwardpass(self, x):
        Z = []
        A = [x]
        n = len(self.weights)
        for i in range(n):
            Ai_hat = np.vstack((A[i], [[1]]))
            Wi_hat = np.hstack((self.weights[i], self.biases[i]))
            Z.append(Wi_hat @ Ai_hat)
            A.append(sigmoid(Z[i]))
        # x_hat = np.vstack((x, [[1]]))
        # W1_hat = np.hstack((self.W1, self.B1))
        # Z2 = W1_hat @ x_hat
        # A2 = sigmoid(Z2)
        # A2_hat = np.vstack((A2, [[1]]))
        # W2_hat = np.hstack((self.W2, self.B2))
        # Z3 = W2_hat @ A2_hat
        # A3 = sigmoid(Z3)
        # return Z2, Z3, A2, A3
        return Z, A

    # def backprop(self, Z2, Z3, A2, A3, x, y):
    def backprop(self, Z, A, x, y):

        Delta = -(y - A[-1]) * sigmoid_derivative(Z[-1])
        for i in range(len(self.weights))[::-1]:
            D_Wi = self.LR * (matrix_mult(Delta, A[i].T)) + self.momentum * self.weights_prev[i]
            D_Bi = self.LR * Delta + self.momentum * self.biases_prev[i]
            self.weights[i] -= D_Wi
            self.biases[i] -= D_Bi
            
            if i > 0 :
                Delta = matrix_mult(self.weights[i].T, Delta) * sigmoid_derivative(Z[i-1])


        self.biases_prev = self.biases.copy()
        self.weights_prev = self.weights.copy()
        # Delta3 = -(y - A3) * sigmoid_derivative(Z3)
        # Delta2 = (self.W2.T @ Delta3) * sigmoid_derivative(Z2)

        # D_W2 = self.LR * (Delta3 @ A2.T) + self.momentum * self.W2_prev
        # D_W1 = self.LR * (Delta2 @ x.T) + self.momentum * self.W1_prev
        # D_B2 = self.LR * (Delta3) + self.momentum * self.B2_prev
        # D_B1 = self.LR * (Delta2) + self.momentum * self.B1_prev
        
        # self.W2 -= D_W2 
        # self.W1 -= D_W1
        # self.B2 -= D_B2
        # self.B1 -= D_B1


def test_basics():
    nn = NeuralNetwork(2, [2], 1, LR = 1)
    # print("actual", nn.weights[0].shape, nn.weights[1].shape)
    nn.weights = [np.ones((2, 2)).astype(np.float64), np.ones((1, 2)).astype(np.float64)]
    # print("goal", nn.weights[0].shape, nn.weights[1].shape)
    nn.weights_prev = [np.zeros((2, 2)).astype(np.float64), np.zeros((1, 2)).astype(np.float64)]
    # print(nn.biases[0].shape, nn.biases[1].shape)
    nn.biases = [np.ones((2, 1)).astype(np.float64), np.array([[1]]).astype(np.float64)]
    # print(nn.biases[0].shape, nn.biases[1].shape)
    # print(type(nn.biases_prev))
    # print(type([np.zeros((2, 1)), np.zeros([[1]])]))
    nn.biases_prev = [np.zeros((2, 1)).astype(np.float64), np.zeros((1, 1)).astype(np.float64)]
    nn.feat_len = 1
    #nn.W1 = np.ones((2, 2))
    # nn.W2 = np.ones((1, 2))
    # nn.B1 = np.ones((2, 1))
    # nn.B2 = np.array([[1.0]])
    x = np.zeros((2, 1)).astype(np.float64)
    y = np.array([1]).astype(np.float64)
    nn.train_singleton(x, y)
    x = np.array([0, 1]).reshape(2,1).astype(np.float64)
    y = np.array([0]).astype(np.float64)
    nn.train_singleton(x, y)

    W1 = np.array([[1.0, 0.99449552], [1.0, 0.99449552]])
    W2 = np.array([[0.95812557, 0.95812557]])
    B2 = 0.9534321554039679
    B1 = np.array([[0.99561354, 0.99561354]])

    assert np.allclose(nn.weights[0], W1, atol=1e-2)
    assert np.allclose(nn.weights[1], W2, atol=1e-2)
    assert np.allclose(nn.biases[0], B1, atol=1e-2)
    assert np.allclose(nn.biases[1], B2, atol=1e-2)

    print("test_basics passed")

if __name__ == "__main__":
    test_basics()
    
