from scipy.io import arff
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as la

class Perceptron:
    """Basic perceptron"""

    # learning_rule = lambda c, t, z, x: c * (t - z) * x

    def __init__(self, learning_rate, input_node_count, bias = 1, f = lambda x: x):
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

    def train(self, data, tol):
        epocs = 0
        err_1 = 0
        err_2 = np.inf

        while epocs < 20:
            for x in data:
                self.learn(x[:-1], float(x[-1]))
            epocs += 1
            err_2 = self.test(data)
            if np.norm(err_1 - err_2) <= tol:
                

    def test(self, data):
        success = 0
        for x in data:
            if self.predict(x[:-1]) == x[-1]:
                success += 1
        return success/len(data[:,0])


def part5():
    dataset, meta = arff.loadarff("voting_task.arff") 
    dataset = dataset.tolist()
    n = len(dataset)
    t = int(n * .7)
    m = len(dataset[0])

    #preprocess
    for i in range(len(dataset)):
        dataset[i] = list(dataset[i])
        for j, y in enumerate(dataset[i]):
            y = y.decode("utf-8") 
            if y == '\'y\'' or y == '\'democrat\'':
                dataset[i][j] = 1
            else:
                dataset[i][j] = 0

    n_epics = 7
    n_splits = 5
    test_set_accuracy = np.zeros((n_splits, n_epics))
    train_set_accuracy = np.zeros((n_splits, n_epics))

    for i in range(n_splits):
        random.shuffle(dataset)
        c_set = np.array(dataset, dtype=int)
        tr_set = c_set[:t, :]
        te_set = c_set[t:, :]
        P = Perceptron(.1, m-1)
        
        for j in range(n_epics):
            P.train(tr_set, 1)
            test_set_accuracy[i, j] = P.test(te_set)
            train_set_accuracy[i, j] = P.test(tr_set)
    


    fig, axs = plt.subplots(2,1)
    collabel = ["split: " + str(x) for x in range(1, n_splits+1)]
    rowlabel = [str(x) + " epocs" for x in range(1, n_epics+1)]
    axs[0].axis('tight')
    axs[0].axis('off')
    the_table = axs[0].table(
            cellText=test_set_accuracy.T,
            colLabels=collabel,
            rowLabels=rowlabel,
            loc='center'
        )
    fig.suptitle('Accuracy', fontsize=16)

    # inacs = np.mean(test_set_accuracy, axis=0)

    # plt.xlabel("Error")
    # plt.plot(range(n_epics), inacs)
    plt.title("Accuracy")
    plt.show()

    # P.test(tr_set)
    # print(P.test(te_set))

part5()






def plot(data, weights, title):
    d2 = np.linspace(-1, 1, 2)
    y2 = [(-weights[2] - x*weights[0])/weights[1] for x in d2]

    plt.scatter(data[4:, 0], data[4:, 1])
    plt.scatter(data[:4, 0], data[:4, 1])
    plt.plot(d2, y2)
    plt.grid(True)
    plt.title(title)
    plt.show()

def part3():
    sep_dataset, meta = arff.loadarff("linearlySeperable.arff")
    non_sep_dataset, meta = arff.loadarff("linearlyUnseperable.arff")
    sep_dataset = sep_dataset.tolist()
    non_sep_dataset = non_sep_dataset.tolist()

    trainingset = np.array(non_sep_dataset + sep_dataset, dtype=np.float)

    P = Perceptron(.1, 2)
    P.train(trainingset, 3)
    
    plot(trainingset, P.weights, "")
    # print(trainingset)


# part3()

def test():
    data, meta = arff.loadarff("linearlySeperable.arff")
    data = np.array(data.tolist(), dtype=np.float)
    P = Perceptron(.1, 2)
    P.train(data, 10)
    plot(data, P.weights, "linearly Seperable")

    data, meta = arff.loadarff("linearlyUnseperable.arff")
    data = np.array(data.tolist(), dtype=np.float)
    P = Perceptron(.1, 2)
    P.train(data, 10)
    plot(data, P.weights, "linearly Seperable")

# test()



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

    # data, meta = arff.loadarff("linearlyUnseperable.arff")
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
    # plt.title("Linearly Unseperable")
    # plt.show()