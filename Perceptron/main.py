#!/usr/bin/env python

import cv2
import os, sys
from os import listdir
from os.path import isfile, join
from Perceptron import Perceptron
import numpy as np
from sklearn.utils import shuffle


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

    print(correct/te_dataset.shape[0])
    # print(P.train(train_data, test_data))


if __name__=="__main__":

    dataset = np.load("image_vecs.npy")
    guess_faces(dataset)
