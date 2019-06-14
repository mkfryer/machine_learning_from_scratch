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
        # print("initial centroids \n", self.centroids, "\n \n \n")

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

    def report(self, cluster_errs):
        print("Total Clusters:", self.k, "\n \n")
        err = 0
        for i in range(self.k):
            centroid = self.centroids[i]
            cluster = self.clusters[i]
            n = len(cluster)
            print("Cluster:", i)
            print("Centroid:", "\n", centroid)
            print("Instances in centroid:", n)
            print("Centroid SSE:", cluster_errs[i])
        print("\nTotal SSE:", sum(cluster_errs), "\n")

    def populate_clusters(self):
            cluster_errs = np.zeros(self.k)
            self.clusters = [[] for x in range(self.k)]
            for i in range(self.m):
                x = self.train_data[i, :-1].copy()
                distances = np.zeros(self.k)
                for j in range(self.k):
                    distances[j] = self.get_sqrd_dist(x, self.centroids[j])
                l = np.argmin(distances)
                self.clusters[l].append(x)
                cluster_errs[l] = cluster_errs[l] +  distances[l]
            return cluster_errs

    def train(self, tol = .0001):
        last_err, cur_err = (np.inf, 0)

        while (cur_err - last_err)**2 > tol:
            last_err = cur_err
            cluster_errs = self.populate_clusters()
            cur_err = sum(cluster_errs)
            self.report(cluster_errs)

            #calculate new centroids
            for i in range(self.k):
                self.centroids[i] = self.calculate_centroid(self.clusters[i])




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
    print(kmc.get_err())
    # print("second centroid \n", kmc.centroids[:, -1])

    # x = np.array([4, 30 , 30, 4])
    # print(x)
    # print(np.bincount(x).argmax())

    # print(kmc.get_sqrd_dist(data[3, :], data[8, :]))
    # print(kmc.get_sqrd_dist(data[1, :], data[8, :]))

    # show_table(
    #     [data[3, :], data[8, :], data[1, :]]
    # )

test_cases()




