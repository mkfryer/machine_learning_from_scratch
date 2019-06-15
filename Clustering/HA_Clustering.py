import numpy as np
from toolkit.arff import Arff
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

class HA_Clustering:

    def __init__(self, k, train_data, test_data, attr_types, a_type = "single link", attr_idx = None):
        """
        """
        self.k = k
        self.train_data = train_data
        self.test_data = test_data
        self.attr_types = attr_types
        self.m, self.n = train_data.shape
        self.attr_idx = attr_idx
        self.clusters = [[train_data[i, :]] for i in range(self.m)] 
        self.clusterize(a_type)

    def get_dist(self, x, y):
        sqrd_dist = 0
        n = len(x)
        for i in range(n):
            if np.isnan(x[i]) or np.isnan(y[i]):
                sqrd_dist += 1
            elif self.attr_types[i] == 'real':
                sqrd_dist += (x[i] - y[i])**2
            elif x[i] != y[i]:
                sqrd_dist += 1
        return np.sqrt(sqrd_dist)

    def get_closest_clusters(self, a_type):
        m = len(self.clusters)
        c_dist = np.zeros((m, m))
        np.fill_diagonal(c_dist, np.inf) 
        #compare all clusters
        for i in range(m):
            for j in range(m):
                if i == j: continue
                m_v = len(self.clusters[i])
                n_v = len(self.clusters[j])
                v_dist = np.zeros((m_v, n_v))
                for k in range(m_v):
                    for l in range(n_v):
                        v_dist[k, l] = self.get_dist(self.clusters[i][k], self.clusters[j][l])
                if a_type == "single link":
                    c_dist[i, j] = np.min(v_dist)
                elif a_type == "complete link":
                    c_dist[i, j] = np.max(v_dist)
        print(c_dist)
        i, j = np.unravel_index(np.argmin(c_dist, axis=None), c_dist.shape)
        print(c_dist[i, j])
        return i, j

    
    def combine_clusters(self, i , j):
        print("Mergin Clusters", i, "and", j)
        print(self.clusters[i], self.clusters[j])

    def clusterize(self, a_type):
        while len(self.clusters) > self.k:
            next_cluster_idxs = self.get_closest_clusters(a_type)
            print(next_cluster_idxs)
            self.combine_clusters(next_cluster_idxs)
            self.k -= 1
            return


    
def test_cases():
    # test_1()

    attr_types = [
        "real",
        "real",
        "real",
        "real",
        "cat",
        "real",
        "cat",
        "real",
        "real",
        "cat",
        "real",
        "cat",
        "cat",
        "cat",
        "cat",
        "cat",
        "cat"
    ]

    attr_idx = [
            [],
            [],
            [],
            [],
            ['none','tcf','tc'],
            [],
            ['none','ret_allw','empl_contr'],
            [],
            [],
            ['yes','no'],
            [],
            ['below_average','average','generous'],
            ['yes','no'],
            ['none','half','full'],
            ['yes','no'],
            ['none','half','full'],
            ['bad','good']
        ]

    k = 5
    arff = Arff("labor.arff")
    # arff.normalize()
    features = arff.get_features().data
    labels = arff.get_labels().data
    # attributes = arff.get_attr_names()
    data = np.hstack((features, labels))[:, 1:]
    kmc = HA_Clustering(k, data, data, attr_types, "single link", attr_idx)

test_cases()