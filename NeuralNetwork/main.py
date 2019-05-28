from toolkit.arff import Arff
import sys
import numpy as np
from NeuralNetwork import NeuralNetwork, test_basics
import matplotlib.pyplot as plt

def main():
    test_basics()

    arff = Arff(sys.argv[1])
    arff.shuffle()
    n = len(arff.get_labels().data)
    t = int(n * .75)
    train_set = arff.create_subset_arff(row_idx=slice(0, t, 1))
    test_set = arff.create_subset_arff(row_idx=slice(t, n, 1))
    test_labels = np.zeros((n-t, 3))
    test_features = test_set.get_features().data
    for i, j in enumerate(test_set.get_labels().data.flatten().astype(int)):
        test_labels[i, j] = 1

    nn = NeuralNetwork(4, 9, [0, 1, 2], LR=.1)
    errs_mse, acc_all_te = nn.train_set(train_set, test_set)

    # d = [x for x in range(0, 100)]
    # r = []
    # for _ in range(100):
    #     train_set.shuffle()
    #     features = train_set.get_features().data
    #     labels = train_set.get_labels().data
    #     train_labels = np.zeros((n, 3))

    #     for i, j in enumerate(labels.flatten().astype(int)):
    #         train_labels[i, j] = 1

    #     for i in range(t):
    #         x = features[i, :].reshape(4,1).astype(float)
    #         y = train_labels[i, :].reshape(3,1).astype(float)
    #         nn.train(x, y)

    #     correct = 0
    #     for i in range(len(test_features)):
    #         x = test_features[i, :].reshape(4,1).astype(float)
    #         y = test_labels[i, :].reshape(3,1).astype(float)
    #         if nn.predict(x) == np.argmax(y):
    #             correct += 1

    #     r.append(1 - (correct/(n - t)))
    d = [x for x in range(len(errs_mse))]
    plt.plot(d, errs_mse)
    plt.plot(d, acc_all_te)
    plt.show()


if __name__ == "__main__":
    main()