import numpy as np
from toolkit.arff import Arff
from scipy.spatial.distance import cdist

class KMC:

    def __init__(self, k, train_data, test_data, attr_types):
        self.k = k
        self.train_data = train_data
        self.test_data = test_data
        self.attr_types = attr_types
        self.m, self.n = train_data.shape
        self.centroids = train_data[:k, :].copy()
        # print("initial centroids", self.centroids, "\n")

    def get_sqrd_dist(self, x, y):
        sqrd_dist = 0
        n = len(x)
        for i in range(n):
            if self.attr_types[i] == 'real':
                if np.isnan(x[i]) or np.isnan(y[i]):
                    sqrd_dist += 1
                else:
                    sqrd_dist += (x[i] - y[i])**2
            else:
                if np.isnan(x[i]) or np.isnan(y[i]):
                    sqrd_dist += 1
                elif x[i] != y[i]:
                    sqrd_dist += 1
        return sqrd_dist

    def calculate_centroid(self, cluster):
        centroid = np.zeros(self.n)
        cluster = np.array(cluster)
        m, n = cluster.shape
        for i in range(n):
            if self.attr_types[i] == 'real':
                x = cluster[:, i]
                x = x[~np.isnan(x)]
                # print("x:", x)
                if len(x) > 0:
                    # print("x", x)
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
            # print("centroid", centroid[i])
        return centroid
    
    # def populate_clusters():

    def train(self, get_distance):

        #populate clusters and calculate new centroids
        clusters = [[] for x in range(self.k)]
        for i in range(self.m):
            x = self.train_data[i, :].copy()
            distances = np.zeros(self.k)
            # print("vector:", x)
            for j in range(self.k):
                # print("cluster:", self.centroids[j])
                distances[j] = get_distance(x, self.centroids[j])
            # print("d", distances, "\n")
            l = np.argmin(distances)
            # print("chosen centroid: ", self.centroids[l])
            clusters[l].append(x)
        
        # print("clusters: \n", clusters, "\n")
        for i in range(self.k):
            self.centroids[i] = self.calculate_centroid(clusters[i])
        
        # print("centroids:", self.centroids)


def test_1():
    data = np.array([
        [.9, .8],
        [.2, .2],
        [.7, .6],
        [-.1, -.6],
        [.5, .5]
    ])
    attr_types = [
        "real",
        "real"
    ]
    k = 2
    get_distance = lambda x, y: cdist([x], [y], metric='cityblock')
    kmc = KMC(k, data, data, attr_types)
    kmc.train(get_distance)
    kmc.train(get_distance)
    centroids_t = np.array([[ 0.7, 0.63333333], [0.05,-0.2]])
    assert(np.allclose(centroids_t, kmc.centroids))
    print("test 1: pass")

def test_cases():
    # test_1()


    attr_types = [
        "real",
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
    k = 5
    arff = Arff("labor.arff")
    features = arff.get_features().data
    labels = arff.get_labels().data
    # attributes = arff.get_attr_names()
    data = np.hstack((features, labels))
    kmc = KMC(k, data, data, attr_types)
    kmc.train(kmc.get_sqrd_dist)

test_cases()