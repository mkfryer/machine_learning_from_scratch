from toolkit.arff import Arff
import sys
import numpy as np
from NeuralNetwork import NeuralNetwork, test_basics
import matplotlib.pyplot as plt

def prob2():
    test_basics()

    arff = Arff(sys.argv[1])
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1))
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1))
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1))

    nn = NeuralNetwork(4, 9, [0, 1, 2], LR=.1)
    all_acc_va, all_mse_va, all_mse_te = nn.train_set(train_set, test_set, validation_set)
 
    d = [x for x in range(len(all_acc_va))]
    plt.plot(d, all_mse_te, label="test MSE")
    plt.plot(d, all_mse_va, label="Val. MSE")
    plt.plot(d, all_acc_va, label="Val. Accuracy")
    plt.title("Iris Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("%")
    plt.legend()
    plt.show()

def prob3():
    """ """
    


if __name__ == "__main__":
    prob2()