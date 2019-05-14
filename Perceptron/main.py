#!/usr/bin/env python

import cv2
import os, sys
from os import listdir
from os.path import isfile, join
from Perceptron import Perceptron
import numpy as np
from sklearn.utils import shuffle
from io import StringIO
import matplotlib.pyplot as plt
import random
from numpy import linalg as la
import tmp_toolkit
from scipy.io import arff


def guess_faces(dataset):
    i_dataset = dataset[:15, :]
    k_dataset = dataset[15:29, :]
    c_dataset = dataset[29:, :]
    ck_dataset = shuffle(np.vstack((c_dataset, k_dataset)))
    ik_dataset = shuffle(np.vstack((i_dataset, k_dataset)))
    c_dataset[:, -1] = np.zeros(16)
    ic_dataset = shuffle(np.vstack((i_dataset, c_dataset)))
    
    ik_tr_dataset = ik_dataset[20:, :]
    ik_te_dataset = ik_dataset[:20, :]
    ck_tr_dataset = ck_dataset[20:, :]
    ck_te_dataset = ck_dataset[:20, :]
    ic_tr_dataset = ic_dataset[21:, :]
    ic_te_dataset = ic_dataset[:21, :]

    P_ik = Perceptron(.1, 35*35)
    P_ic = Perceptron(.1, 35*35)
    P_ck = Perceptron(.1, 35*35)

    te_dataset = np.vstack((ik_te_dataset, ic_te_dataset, ck_te_dataset))
    correct = 0
    for x in te_dataset:
        output1, net1 = P_ik.predict(x)
        output2, net2 = P_ic.predict(x)
        output3, net3 = P_ck.predict(x)
        d = { net1:output1, net2:output2, net3:output3 }

        if d[max(net1, net2, net3)] == x[-1]:
            correct += 1

    print("3 class image predictor success rate: ", 100 * correct/te_dataset.shape[0], "%")
    # print(P.train(train_data, test_data))

def show_table(collabel, rowlabel, m):
    fig, axs = plt.subplots(1,1)
    axs.axis('tight')
    axs.axis('off')
    axs.table(
            cellText=m,
            colLabels=collabel,
            rowLabels=rowlabel,
            loc='center'
        )
    plt.show()

def part_3():
    sep_dataset, meta = arff.loadarff("linearlySeperable.arff")
    non_sep_dataset, meta = arff.loadarff("linearlyUnseperable.arff")
    sep_dataset = sep_dataset.tolist()
    non_sep_dataset = non_sep_dataset.tolist()
    trainingset = np.array(non_sep_dataset + sep_dataset, dtype=np.float)

    learning_rates = [.00001, .001, .1, .2, .5, .8, .99, .99999]
    m = np.zeros((len(learning_rates), 3))

    for i, rate in enumerate(learning_rates):
        P = Perceptron(rate, 2)
        accuracy, epochs = P.train(trainingset, trainingset)
        m[i, :] = np.array([rate, accuracy, epochs])

    collabel = ["Learning Rate", "Accuracy", "Epochs"]
    rowlabel = ["" for x in range(len(learning_rates))]
    show_table(collabel, rowlabel, m)

def part_4():
    ls_data, meta = arff.loadarff("linearlySeperable.arff")
    ls_data = np.array(ls_data.tolist(), dtype=np.float)
    P = Perceptron(.1, 2)
    print(P.train(ls_data, ls_data))
    d2 = np.linspace(-1, 1, 2)
    y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]

    plt.scatter(ls_data[4:, 0], ls_data[4:, 1])
    plt.scatter(ls_data[:4, 0], ls_data[:4, 1])
    plt.plot(d2, y2)
    plt.grid(True)
    plt.title("Linearly Seperable")
    plt.show()

    lu_data, meta = arff.loadarff("linearlyUnseperable.arff")
    lu_data = np.array(lu_data.tolist(), dtype=np.float)
    P = Perceptron(.1, 2)
    P.train(lu_data, lu_data)
    d2 = np.linspace(-1, 1, 2)
    y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]
    lu_data = np.array(lu_data, dtype=np.float)

    plt.scatter(lu_data[4:, 0], lu_data[4:, 1])
    plt.scatter(lu_data[:4, 0], lu_data[:4, 1])
    plt.plot(d2, y2)
    plt.grid(True)
    plt.title("Linearly Unseperable")
    plt.show()

    data = np.vstack((ls_data, lu_data))
    P = Perceptron(.1, 2)
    P.train(data, data)
    d2 = np.linspace(-1, 1, 2)
    y2 = [(-P.weights[2] - x*P.weights[0])/P.weights[1] for x in d2]
    data = np.array(data, dtype=np.float)

    plt.scatter(data[8:, 0], data[8:, 1])
    plt.scatter(data[:8, 0], data[:8, 1])
    plt.plot(d2, y2)
    plt.grid(True)
    plt.title("Linearly Seperable Mixed with Unseperable")
    plt.show()


def part_5():
    dataset = tmp_toolkit.prepare_voting_data()
    n_splits = 5
    info_m = np.zeros((n_splits, 4))
    accuracy_m = np.zeros((n_splits, 26))
    n = len(dataset)
    t = int(n * .7)
    m = len(dataset[0])

    for i in range(n_splits):
        random.shuffle(dataset)
        c_set = np.array(dataset, dtype=int)
        tr_set = c_set[:t, :]
        te_set = c_set[t:, :]
        P = Perceptron(.1, m-1)
        
        te_accuracy, epochs = P.train(tr_set, te_set)
        tr_accuracy = 1 - P.test(tr_set)
        info_m[i] = np.array([te_accuracy[-1], tr_accuracy, (te_accuracy[-1]+tr_accuracy)/2, epochs])
        accuracy_m[i, :] = np.hstack((te_accuracy, np.zeros(26 - len(te_accuracy))))

    collabel = ["Test Set Accuracy", "Training Set Accuracy","Average Accuracy", "Epochs"]
    rowlabel = ["split:" + str(x)  for x in range(1, n_splits + 1)]
    show_table(collabel, rowlabel, np.round(info_m, decimals=5))

    plt.plot(range(20), 1 - np.mean(accuracy_m[:, :20], axis=0))
    plt.title("Voting Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Misclassification Rate")
    plt.show()

def part_6(dataset):
    guess_faces(dataset)



if __name__=="__main__":

    dataset = np.load("image_vecs.npy")

    # part_3()
    # part_4()
    # part_5()
    part_6(dataset)