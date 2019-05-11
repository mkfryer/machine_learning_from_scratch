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

    def train(self, tr_data, te_data, tol=1.23E-7, max_epocs = 15):
        epocs = 0
        prev_err = 0
        next_err = 0

        while True:
            epocs += 1
            prev_err = next_err
            for x in tr_data:
                self.learn(x[:-1], float(x[-1]))
            next_err = self.test(te_data)
            if la.norm(next_err - prev_err) <= tol or epocs >= max_epocs:
                break

        accuracy = 1 - next_err
        return accuracy, epocs

    def test(self, data):
        success = 0
        for x in data:
            prediction, net = self.predict(x[:-1])
            if prediction == x[-1]:
                success += 1
        err = 1 - success/len(data[:,0])
        return err


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

    n_splits = 5
    accuracy_m = np.zeros((n_splits + 1, 4))

    for i in range(n_splits):
        random.shuffle(dataset)
        c_set = np.array(dataset, dtype=int)
        tr_set = c_set[:t, :]
        te_set = c_set[t:, :]
        P = Perceptron(.1, m-1)
        
        te_accuracy, epochs = P.train(tr_set, te_set)
        tr_accuracy = 1 - P.test(tr_set)
        accuracy_m[i] = np.array([te_accuracy, tr_accuracy, (te_accuracy+tr_accuracy)/2, epochs])
        print(P.weights)
    accuracy_m[n_splits] = np.mean(accuracy_m, axis=0)

    fig, axs = plt.subplots(2,1)
    collabel = ["Test Set Accuracy", "Training Set Accuracy","Average Accuracy", "Epochs"]
    rowlabel = ["split:" + str(x)  for x in range(1, n_splits + 2)]
    rowlabel[-1] = "Avg:"
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(
            cellText=np.round(accuracy_m, decimals=5),
            colLabels=collabel,
            rowLabels=rowlabel,
            loc='center'
        )

    axs[1].plot(accuracy_m[:-1, 3], accuracy_m[:-1, 2])
    axs[1].set_title("asdfsa")
    axs[1].set_xlabel("epochs")
    fig.suptitle('Accuracy', fontsize=16)

    plt.show()


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