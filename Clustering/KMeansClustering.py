import numpy as np
from toolkit.arff import Arff
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
class KMC:

    def __init__(self, k, train_data, test_data, attr_types):
        self.k = k
        self.train_data = train_data
        self.test_data = test_data
        self.attr_types = attr_types
        self.m, self.n = train_data.shape
        self.centroids = train_data[:k, :-1].copy()

    def get_sqrd_dist(self, x, y):
        sqrd_dist = 0
        n = len(x)
        for i in range(n):
            if np.isnan(x[i]) or np.isnan(y[i]):
                sqrd_dist += 1
            elif self.attr_types[i] == 'real':
                sqrd_dist += (x[i] - y[i])**2
            elif x[i] != y[i]:
                sqrd_dist += 1
        return sqrd_dist

    def calculate_centroid(self, cluster):
        centroid = np.zeros(self.n -1)
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
    
    def predict(self, x):
        distances = np.zeros(self.k)
        for j in range(self.k):
            distances[j] = self.get_sqrd_dist(x, self.centroids[j])
        return np.argmin(distances)

    def get_accuracy(self):
        m, n = self.test_data.shape
        err = 0
        for i in range(m):
            y = self.predict(self.test_data[i, :-1])
            y_hat = self.test_data[i, -1]
            err += self.get_sqrd_dist([y], [y_hat])
        return err

    def report(self):
        print("Total Clusters:", self.k, "\n")
        err = 0
        for i in range(self.k):
            centroid = self.centroids[i]
            cluster = self.clusters[i]
            n = len(cluster)
            c_err = 0
            for j in range(n):
                c_err += self.get_sqrd_dist(cluster[j], centroid)
            err += c_err
            print("Cluster:", i)
            print("Centroid:", "\n", centroid)
            print("Instances in centroid:", n)
            print("Centroid SSE:", c_err)
        print("Total SSE:", err, "\n \n \n")
            
    def train(self, tol = .0001):
        last_err, cur_err = (np.inf, 0)
        clusters_err = []

        while (cur_err - last_err)**2 > tol:
            last_err = cur_err
            cur_err = 0

            #populate clusters
            self.clusters = [[] for x in range(self.k)]
            for i in range(self.m):
                x = self.train_data[i, :-1].copy()
                distances = np.zeros(self.k)
                for j in range(self.k):
                    distances[j] = self.get_sqrd_dist(x, self.centroids[j])
                l = np.argmin(distances)
                cur_err += distances[l]
                self.clusters[l].append(x)

            #calculate new centroids
            for i in range(self.k):
                self.centroids[i] = self.calculate_centroid(self.clusters[i])

            self.report()


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

def show_table(m, title = ""):
    fig, axs = plt.subplots(1,1)
    axs.axis('tight')
    axs.axis('off')
    axs.table(
            cellText=m,
            loc='center'
        )
    axs.set_title(title)
    plt.show()


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
    k = 5
    arff = Arff("labor.arff")
    features = arff.get_features().data
    labels = arff.get_labels().data
    # attributes = arff.get_attr_names()
    data = np.hstack((features, labels))[:, 1:]
    kmc = KMC(k, data, data, attr_types)
    kmc.train(tol=0)


test_cases()




