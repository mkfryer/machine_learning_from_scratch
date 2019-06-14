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
        print("initial centroids \n", self.centroids, "\n")

    def get_sqrd_dist(self, x, y):
        sqrd_dist = 0
        n = len(x)
        # print("x", x)
        # print("y", y)
        for i in range(n):
            if np.isnan(x[i]) or np.isnan(y[i]):
                # print("nan", x[i], y[i])
                sqrd_dist += 1
            elif self.attr_types[i] == 'real':
                # print("real", x[i], y[i])
                sqrd_dist += (x[i] - y[i])**2
            elif x[i] != y[i]:
                # print("cat not equal", x[i], y[i])
                sqrd_dist += 1
            # else:
                # print('Cat equal', x[i], y[i])

            # print("dist: ", sqrd_dist)
        return sqrd_dist

    def calculate_centroid(self, cluster):
        centroid = np.zeros(self.n -1)
        cluster = np.array(cluster)
        m, n = cluster.shape
        for i in range(n):
            if self.attr_types[i] == "real":
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
                # print("x", x)
                x = x[~np.isnan(x)].astype(int)
                if len(x) > 0:
                    # print("argmax", np.bincount(x).argmax())
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
            x = self.train_data[i, :-1].copy()
            distances = np.zeros(self.k)
            # print("vector:", x)
            for j in range(self.k):
                # print("cluster:", self.centroids[j])
                distances[j] = get_distance(x, self.centroids[j])
            # print("d", distances, "\n")
            l = np.argmin(distances)
            # print("chosen centroid: ", self.centroids[l])
            clusters[l].append(x)
            # print(i, "=", l)
        
        # print("clusters: \n", clusters, "\n")
        for i in range(self.k):
            print("c", i, "\n")
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


# @attribute 'duration' real
# @attribute 'wage-increase-first-year' real
# @attribute 'wage-increase-second-year' real
# @attribute 'wage-increase-third-year' real
# @attribute 'cost-of-living-adjustment' {'none','tcf','tc'}
# @attribute 'working-hours' real
# @attribute 'pension' {'none','ret_allw','empl_contr'}
# @attribute 'standby-pay' real
# @attribute 'shift-differential' real
# @attribute 'education-allowance' {'yes','no'}
# @attribute 'statutory-holidays' real
# @attribute 'vacation' {'below_average','average','generous'}
# @attribute 'longterm-disability-assistance' {'yes','no'}
# @attribute 'contribution-to-dental-plan' {'none','half','full'}
# @attribute 'bereavement-assistance' {'yes','no'}
# @attribute 'contribution-to-health-plan' {'none','half','full'}
# @attribute 'class' {'bad','good'}

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
    kmc.train(kmc.get_sqrd_dist)
    # print("first centroid \n", kmc.centroids[:, -1])
    kmc.train(kmc.get_sqrd_dist)
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




