import numpy as np
from toolkit.arff import Arff
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

class HA_Clustering:

    def __init__(self, k, train_data, test_data, attr_types, a_type = "single link", attr_idx = None):
        """
        """
        self.k = k
        self.train_data = train_data[:, :]
        self.test_data = test_data
        self.attr_types = attr_types
        self.m, self.n = train_data.shape
        self.attr_idx = attr_idx
        self.clusters = [[self.train_data[i, :]] for i in range(self.m)] 
        self.clusterize(a_type)
        self.centroids = []

    def calculate_centroid(self, cluster):
        centroid = np.zeros(self.n)
        cluster = np.array(cluster)
        m, n = cluster.shape
        for i in range(n):
            if self.attr_types[i] == "real":
                x = cluster[:, i]
                x = x[~np.isnan(x)]
                if len(x) > 0:
                    centroid[i] = np.mean(x)
                else:
                    centroid[i] = np.nan
            else:
                x = cluster[:, i]
                x = x[~np.isnan(x)].astype(int)
                if len(x) > 0:
                    centroid[i] = np.bincount(x).argmax()
                else:
                    centroid[i] = np.nan
        return centroid

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
                if c_dist[i, j] != 0: continue
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
                c_dist[j, i] = c_dist[i, j]
        # print(c_dist[28, 32], c_dist[32, 28])
        i, j = np.unravel_index(np.argmin(c_dist, axis=None), c_dist.shape)
        smallest_dist = c_dist[i, j]
        #create tie preference
        mask = np.where(c_dist == smallest_dist)
        print("Merging Clusters", i, "and", j, "Distance:", c_dist[i, j])
        return (i, j)

    
    def combine_clusters(self, i , j):
        self.clusters[i] = self.clusters[i] + self.clusters[j]
        del self.clusters[j]

    def clusterize(self, a_type):
        iterations = -1
        while len(self.clusters) > self.k:
            iterations += 1
            print("************")
            print("Iteration", iterations)
            print("************")
            i, j = self.get_closest_clusters(a_type)
            self.combine_clusters(i, j)
            m = len(self.clusters)
            for i in range(m):
                c = self.clusters[i]
                n = len(c)
                print("cluster", i, "size:", n)
        
        for i in range(self.k):
            centroid = self.calculate_centroid(self.clusters[i])
            centroid = list(np.round(centroid, decimals = 3))
            for j in range(len(centroid)):
                if np.isnan(centroid[j]):
                    centroid[j] = "?"
                elif self.attr_idx and self.attr_types[j] != 'real':
                    centroid[j] = self.attr_idx[j][int(centroid[j])]
            print("Centroid", i, centroid)

    def get_SSE(self):
        SSE = 0
        for i in range(self.k):
            cluster = self.clusters[i]
            for x in cluster:
                SSE += self.get_dist(x, self.calculate_centroid(self.clusters[i]))

        return SSE

    def get_silhouette(self, x, idx):
        def f(x, i):
            count = 0
            dist = 0
            for y in self.clusters[i]:
                if sum(x - y) == 0:
                    continue
                dist += cdist([x], [y], 'cityblock')[0][0]##self.get_dist(x, y)
                count += 1
            if count == 0:
                return 0
            return dist/count
        
        D = np.zeros(self.k)
        for i in range(self.k):
            if i == idx:
                D[i] = np.inf
            else:
                D[i] = f(x, i)
        b = min(D)
        a = f(x, idx)
        # print(x, "a", a, "b", b)

        # print("s", (b - a)/max(a, b))
        return (b - a)/max(a, b)

    def get_global_silhouette(self):

        # self.clusters = [
        #     np.array([[.8, .7], [.9, .8]]),
        #     np.array([[.6, .6], [0, .2], [.2, .1]])
        # ]
        # self.k = 2
        global_s = []
        for i in range(self.k):
            for j in range(len(self.clusters[i])):
                global_s.append(self.get_silhouette(self.clusters[i][j], i))
        # print("total", sum(global_s)/len(global_s))
        return sum(global_s)/len(global_s)



    
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
    data = np.hstack((features, labels))[:, 1:-1]
    kmc = HA_Clustering(k, data, data, attr_types, "complete link", attr_idx)

test_cases()