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
    all_acc_va, all_mse_va, all_mse_te, all_mse_tr = nn.train_set(train_set, test_set, validation_set)
 
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

    best_mse_te = []
    best_mse_tr = []
    best_mse_va = []
    epochs = []

    LRS = [.01, .1, .5, .8, 1.5]
    for LR in LRS:
        # print(LR)
        nn = NeuralNetwork(8, [16], 11, LR=LR, momentum=0)
        all_acc_va, all_mse_va, all_mse_te, all_mse_tr = nn.train_set(train_set, test_set, validation_set, w = 5)
        best_mse_te.append(min(all_mse_te))
        best_mse_tr.append(min(all_mse_tr))
        best_mse_va.append(min(all_mse_va))
        epochs.append(len(all_mse_va))

    plt.plot(LRS, best_mse_te, label="MSE Te")
    plt.plot(LRS, best_mse_tr, label="MSE Tr")
    plt.plot(LRS, best_mse_va, label="MSE V.A")
    plt.title("Vowel MSE vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    plt.plot(LRS, epochs)
    plt.title("Vowel Epochs vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.legend()
    plt.show()

def prob4():
    arff = Arff(sys.argv[2])
    imp_atts = [1, 3, 4, 5, 7, 9, 11, 12, 13]
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1), col_idx = imp_atts)
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1), col_idx = imp_atts)
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1), col_idx = imp_atts)

    best_mse_te = []
    best_mse_tr = []
    best_mse_va = []
    hidden_nodes = [1, 3, 6, 10, 13, 15, 16, 18, 20, 22, 25, 30, 40]

    for nodes in hidden_nodes:
        # print(nodes)
        nn = NeuralNetwork(8, [nodes], 11, LR = .1, momentum=0)
        all_acc_va, all_mse_va, all_mse_te, all_mse_tr = nn.train_set(train_set, test_set, validation_set, w = 5)
        
        best_mse_te.append(min(all_mse_te))
        best_mse_tr.append(min(all_mse_tr))
        best_mse_va.append(min(all_mse_va))

    plt.plot(hidden_nodes, best_mse_te, label="MSE Te")
    plt.plot(hidden_nodes, best_mse_tr, label="MSE Tr")
    plt.plot(hidden_nodes, best_mse_va, label="MSE V.A")
    plt.title("Vowel MSE vs Hidden Nodes")
    plt.xlabel("Hidden Nodes")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def prob5():
    arff = Arff(sys.argv[2])
    imp_atts = [1, 3, 4, 5, 7, 9, 11, 12, 13]
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1), col_idx = imp_atts)
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1), col_idx = imp_atts)
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1), col_idx = imp_atts)

    epochs = []
    momentums = np.linspace(0, 1.5, 20)
    # momentums = [.5, 1]

    for momentum in momentums:
        print(momentum)
        nn = NeuralNetwork(8, [30], 11, LR = .1, momentum=momentum)
        all_acc_va, all_mse_va, all_mse_te, all_mse_tr = nn.train_set(train_set, test_set, validation_set, w = 5)
        epochs.append(len(all_acc_va))

    plt.plot(momentums, epochs)
    plt.title("Vowel Momentum vs Epoch Convergence")
    plt.xlabel("Momentum")
    plt.ylabel("Epochs til Conv.")
    plt.show()

def prob6():
    arff = Arff(sys.argv[1])
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .55)
    v = n - int(n * .20)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1))
    test_set = arff.create_subset_arff(row_idx=slice(t, v, 1))
    validation_set = arff.create_subset_arff(row_idx=slice(v, n, 1))

    nn = NeuralNetwork(4, [9, 8, 8, 7, 9, 8], 3, LR=.01)
    all_acc_va, all_mse_va, all_mse_te, all_mse_tr = nn.train_set(train_set, test_set, validation_set)
 
    d = [x for x in range(len(all_acc_va))]
    plt.plot(d, all_mse_te, label="test MSE")
    plt.plot(d, all_mse_va, label="Val. MSE")
    plt.plot(d, all_acc_va, label="Val. Accuracy")
    plt.title("Iris Dataset W/ 6 hidden layers")
    plt.xlabel("Epochs")
    plt.ylabel("%")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_basics()
    prob2()
    prob3()
    prob4()
    prob5()
    prob6()