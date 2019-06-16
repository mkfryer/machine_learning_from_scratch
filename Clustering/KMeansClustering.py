import numpy as np
from toolkit.arff import Arff
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt



class KMC:

    def __init__(self, k, train_data, test_data, attr_types, attr_idx = None):
        self.k = k
        self.train_data = train_data[:, :]
        self.test_data = test_data
        self.attr_types = attr_types
        self.m, self.n = train_data.shape
        self.attr_idx = attr_idx
        self.centroids = train_data[:k, :].copy()
        
    def inhance_centroids(self):
        def P(i):
            p = 0
            for j in range(self.m):
                if self.get_sqrd_dist(self.train_data[i, :], self.train_data[j, :]) < 1.3:
                    p += 1
            return p
        densities = np.zeros(self.m)
        for i in range(self.m):
            densities[i] = P(i)
        idx_mask = []

        for idx in np.argsort(densities)[::-1]:
            if len(idx_mask) == 0:
                idx_mask.append(idx)
            for idx2 in idx_mask:
                if self.get_sqrd_dist(self.train_data[idx2, :], self.train_data[idx, :]) < 1:
                    continue 
            idx_mask.append(idx)

        self.centroids = self.train_data[idx_mask, :]
        print(self.train_data[idx_mask, :])

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
    
    def predict(self, x):
        distances = np.zeros(self.k)
        for j in range(self.k):
            distances[j] = self.get_sqrd_dist(x, self.centroids[j])
        return np.argmin(distances)

    # def get_accuracy(self):
    #     m, n = self.test_data.shape
    #     err = 0
    #     for i in range(m):
    #         y = self.predict(self.test_data[i, :-1])
    #         y_hat = self.test_data[i, -1]
    #         err += self.get_sqrd_dist([y], [y_hat])
    #     return err

    def report(self, cluster_errs, iteration):
        print("******************************")
        print("Iteration", iteration)
        print("******************************")
        err = 0
        for i in range(self.k):
            centroid = list(np.round(self.centroids[i], decimals = 3))
            for j in range(len(centroid)):
                if np.isnan(centroid[j]):
                    centroid[j] = "?"
                elif self.attr_idx and self.attr_types[j] != 'real':
                    centroid[j] = self.attr_idx[j][int(centroid[j])]
            cluster = self.clusters[i]
            n = len(cluster)
            print("Cluster:", i, "Size:", n, "SSE:", cluster_errs[i])
            print("Centroid:", i, "=" , centroid)
        print("\nTotal Clusters:", self.k , "\nTotal SSE:", np.round(sum(cluster_errs), decimals = 3), "\n \n")


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

    def train(self, report = True, tol = .0001):
        last_err, cur_err = (np.inf, 0)
        iteration = 0
        while (cur_err - last_err)**2 > tol:
            iteration += 1
            last_err = cur_err
            cluster_errs = self.populate_clusters()
            cur_err = sum(cluster_errs)

            if report:
                self.report(cluster_errs, iteration)

            #calculate new centroids
            for i in range(self.k):
                self.centroids[i] = self.calculate_centroid(self.clusters[i])
        
        if report:
            print("SSE has converged")
        return cur_err

    def get_silhouette(self, x, idx):
        def f(x, i):
            count = 0
            dist = 0
            for y in self.clusters[i]:
                if np.allclose(x, y):
                    continue
                dist += cdist([x], [y], 'cityblock')[0][0]#self.get_sqrd_dist(x, y)
                count += 1
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
    arff.normalize()
    features = arff.get_features().data
    labels = arff.get_labels().data
    # attributes = arff.get_attr_names()
    data = np.hstack((features, labels))[:, 1:]
    kmc = KMC(k, data, data, attr_types, attr_idx)
    kmc.train(tol=0)

# test_cases()




