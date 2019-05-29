from toolkit.arff import Arff
import sys
import numpy as np
from NeuralNetwork import NeuralNetwork, test_basics
import matplotlib.pyplot as plt

def prob2():
    arff = Arff(sys.argv[1])
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1))
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1))
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1))

    nn = NeuralNetwork(4, [9], 3, LR=.1)
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
    arff = Arff(sys.argv[2])
    imp_atts = [1, 3, 4, 5, 7, 9, 11, 12, 13]
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1), col_idx = imp_atts)
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1), col_idx = imp_atts)
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1), col_idx = imp_atts)
    nn = NeuralNetwork(8, [16, 14], 11, LR=.01, momentum=0)
    all_acc_va, all_mse_va, all_mse_te = nn.train_set(train_set, test_set, validation_set, w = 5)

    d = [x for x in range(len(all_acc_va))]
    plt.plot(d, all_mse_te, label="test MSE")
    plt.plot(d, all_mse_va, label="Val. MSE")
    plt.plot(d, all_acc_va, label="Val. Accuracy")
    plt.title("Vowel Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("%")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_basics()
    prob3()