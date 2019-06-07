"""
"""
import numpy as np


class KNNClassifier:

    def __init__(self, k, training_data, testing_data):
        """
        """
        self.k = k
        self.training_data = training_data
        self.testing_data = testing_data
        self.tr_m, self.tr_n = training_data.shape
        self.te_m, self.te_n = testing_data.shape

    def predict(self, x):
        M = self.training_data[:, :-1] - x
        D = np.linalg.norm(M, axis=1)
        #sort data and get indicies
        nn_idx = np.argpartition(D, 3)[:self.k]
        #get corresponding labels
        nn_labels = self.training_data[nn_idx, -1].astype(int)
        #get most common label
        p_label = np.bincount(nn_labels).argmax()
        return p_label

    def get_accuracy(self):
        correct = 0
        for x in self.testing_data:
            if np.allclose(self.predict(x[:-1]), x[-1]):
                correct += 1
        return correct/self.te_m

def test():

    KNNC = KNNClassifier()