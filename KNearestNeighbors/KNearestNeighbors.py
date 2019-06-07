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
        # self.get_weighted_d = lambda x, y: 1/np.linalg.norm(x - y)**2

    def regress_predict(self, x, weighted_d):
        M = self.training_data[:, :-1] - x
        D = np.linalg.norm(M, axis=1)
        #sort data and get indicies
        nn_idx = np.argpartition(D, 3)[:self.k]

        if weighted_d == False:
            return sum(self.training_data[nn_idx, -1])/self.k
        else:
            nn = self.training_data[nn_idx, :-1]
            f_hat = self.training_data[nn_idx, -1]
            w = 1 / np.linalg.norm(x - nn, axis=1)**2
            return np.sum(w * f_hat)/np.sum(w)

    def get_accuracy_regress(self, weighted_d = False):
        mse = 0
        for x in self.testing_data:
            x_hat = self.regress_predict(x[:-1], weighted_d)
            mse += (x[-1] - x_hat)**2
        return mse/self.te_m

    def predict(self, x, weighted_d):
        
        M = self.training_data[:, :-1] - x
        D = np.linalg.norm(M, axis=1) 
        #sort data and get indicies
        nn_idx = np.argpartition(D, 3)[:self.k]
        #get corresponding labels
        nn_labels = self.training_data[nn_idx, -1].astype(int)
        #get most common label
        if weighted_d == False:
            return np.bincount(nn_labels).argmax()
        else:
            #todo - clean this up ... yuck
            W = 1 / np.linalg.norm(x - self.training_data[nn_idx, :-1], axis=1)**2
            uniq_classes = np.unique(self.training_data[nn_idx, -1])
            index_scores = []
            for class_i in uniq_classes:
                i_mask = np.where(self.training_data[nn_idx, -1] == class_i)[0]
                index_scores.append(np.sum(W[i_mask]))
            b_class_i = np.argmax(np.array(index_scores))
            return uniq_classes[b_class_i]
        

    def get_accuracy(self, weighted_d = False):
        correct = 0
        for x in self.testing_data:
            if np.allclose(self.predict(x[:-1], weighted_d), x[-1]):
                correct += 1
        return correct/self.te_m

def test():

    KNNC = KNNClassifier()