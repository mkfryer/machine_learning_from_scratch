"""
Decision Tree
"""
import numpy as np

class Node:

    def __init__(self, data):
        # self.children = children
        self.data = data
    

class DecisionTree:
    def __init__(self, data):
        """

        """
        self.data = data
        self.root = None
    
    def calc_entropy(self, targ_col_data):
        uniq = np.unique(targ_col_data)
        n = len(targ_col_data)
        p = []
        for x in uniq:
            m = np.where(targ_col_data == x)[0]
            p.append(len(m)/n)
        p = np.array(p)
        return -p @ np.log2(p)
    
    def calc_gain(self, data, attr_index):
        """ """
        # subdata = np.vstack((self.data[:, attr_index], self.data[:, -1])).T
        # subdata = subdata[row_mask]
        entr_global = self.calc_entropy(self.data[:, -1])
        sub_entr = []
        vals = np.unique(data[:, attr_index])

        for val in vals:
            m = np.where(data == val)[0]
            sub_entr.append((len(m)/len(data[:, attr_index])) * self.calc_entropy(data[:, -1][m]))

        gain = entr_global - sum(sub_entr)
        return gain

    def find_max_gain(self, data):
        m, n = data.shape
        #ignore last column since it is the category
        gain = np.zeros(n-1)
        for i in range(n-1):
            gain[i] = self.calc_gain(data, i)
        max_gain_index = np.argmax(gain)

        return max_gain_index

    # def learn(self, data):
    #     i = find_max_gain(data)
    #     self.root = Node(data[])

if __name__=="__main__":
    print(
        # (1/3) * np.log2(1/3) + (2/3) * np.log2(2/3)
        # .919 + np.log2(1/3)
        # (2/9) * np.log2(1/2) - .113
        (2/9) * np.log(1)
    )


    a = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"]
    b = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"]
    c = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
    d = ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "strong"]
    e =["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes" , "yes", "yes", "no"]

    f = [a, b, c, d, e]
    # keys = set()
    # for x in f:
    #     for y in x:
    #         keys.add(y)
    # print(keys)

    keys = ["sunny", "rainy", "overcast", "hot", "mild", "cool", "high", "normal", "weak", "strong", "no", "yes"]
    d_k = {key:x for x,key in enumerate(keys)}
    cols = ["outlook", "temperature", "humidity" ,"wind", "playtennis"]
    k_c = {key:x for x,key in enumerate(cols)}
    data = np.zeros((14, 5))

    for j, x in enumerate(f):
        for i, y in enumerate(x):
            data[i, j] = d_k[y]

    # print(data)
    # print(k_c)
    a = np.array([
        [1, 2, 5, 7],
        [0, 3, 5, 8],
        [0, 4, 6, 9],
        [1, 4, 6, 7],
        [1, 3, 5, 9],
        [1, 3, 6, 7],
        [0, 2, 6, 9],
        [1, 3, 5, 9],
        [0, 2, 5, 8],
        ])
    print(a)
    T = DecisionTree(a)
    print(T.calc_gain(a, 0))
    print(T.calc_gain(a, 1))
    print(T.calc_gain(a, 2))