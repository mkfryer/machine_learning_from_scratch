"""
Decision Tree
"""
import numpy as np

class Branch:

    def __init__(self, attr_idx, cat_idx):
        self.attr_idx = attr_idx
        self.cat_idx = cat_idx
        self.c_node = None

    def add_connection(self, node):
        self.c_node = node

    # def show(self):
    #     if self.c_node != None:
    #         print(self.c_node.get_class())
    #         for branch in self.c_node.branches:
    #             branch.show()

class Node:

    def __init__(self, data, attr_idx):
        # self.children = children
        self.data = data
        self.branches = []
        self.attr_idx = attr_idx

    def connect_branch(self, branch):
        self.branches.append(branch)

    def get_class(self):
        cols = ["outlook", "temperature", "humidity" ,"wind", "playtennis"]
        print(cols[self.attr_idx])
        for branch in self.branches:
            branch.c_node.get_class()
        print("exausted branches")



class DecisionTree:
    def __init__(self, data):
        """
        """
        self.data = data
        self.m, self.n = data.shape
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
        m, n = data.shape
        entr_global = self.calc_entropy(self.data[:, -1])
        sub_entr = []
        categories = np.unique(data[:, attr_index])
       
        for cat in categories:
            cat_mask = np.where(data == cat)[0]
            sub_entr.append((len(cat_mask)/m) * self.calc_entropy(data[:, -1][cat_mask]))

        gain = entr_global - sum(sub_entr)
        return gain

    def find_max_gain(self, data, attrs_idx_states):
        n = len(attrs_idx_states)
        gains = np.zeros(n)
        active_attr_indxs, = np.where(attrs_idx_states == 1)
        for attr_idx in active_attr_indxs:
            gains[attr_idx] = self.calc_gain(data, attr_idx)
        max_gain_index = np.argmax(gains)

        return max_gain_index

    def split_data(self, data, i):
        """
        i - index of column(attr) to split on
        """
        attributes = np.unique(data[:, i])
        splits = []
        for attr in attributes:
            idx_mask, = np.where(data[:, i] == attr)
            splits.append(data[idx_mask, :])
        return splits
        
    def start_learn(self, data):
        #attributes to split on. Can split if attr_indx = 1, 0 means you cant use it
        attrs_idx_states = np.ones(self.n)
        #dont split on category
        attrs_idx_states[-1] = 0 
        #index corresponding to best gain
        bst_attr_idx = self.find_max_gain(data, attrs_idx_states)
        self.root = Node(data, bst_attr_idx)

        data_attr_splits = self.split_data(data, bst_attr_idx)
        # avail_attrs_idxs = np.delete(avail_attrs_idxs, [bst_attr_idx], None)
        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        for i, data_split in enumerate(data_attr_splits):
            branch = Branch(bst_attr_idx, i)
            self.root.connect_branch(branch)
            self.learn(data_split, branch, attrs_idx_states.copy())

    def learn(self, data, branch, attrs_idx_states):
        bst_attr_idx = self.find_max_gain(data, attrs_idx_states)
        child = Node(data, bst_attr_idx)
        branch.add_connection(child)

        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        data_attr_splits = self.split_data(data, bst_attr_idx)

        for i, data_split in enumerate(data_attr_splits):
            #base case - learn only when data_split is not pure
            if len(np.unique(data[:, -1])) > 1:
                branch = Branch(bst_attr_idx, i)
                child.connect_branch(branch)
                self.learn(data_split, branch, attrs_idx_states.copy())

    def show_tree(self):
        """ """
        self.root.get_class()

if __name__ == "__main__":
    # print(
    #     # (1/3) * np.log2(1/3) + (2/3) * np.log2(2/3)
    #     # .919 + np.log2(1/3)
    #     # (2/9) * np.log2(1/2) - .113
    #     (2/9) * np.log(1)
    # )


    a = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"]
    b = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"]
    c = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
    d = ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "strong"]
    e = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes" , "yes", "yes", "no"]

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
    # print(data)
    # print(k_c)
    # a = np.array([
    #     [1, 2, 5, 7],
    #     [0, 3, 5, 8],
    #     [0, 4, 6, 9],
    #     [1, 4, 6, 7],
    #     [1, 3, 5, 9],
    #     [1, 3, 6, 7],
    #     [0, 2, 6, 9],
    #     [1, 3, 5, 9],
    #     [0, 2, 5, 8],
    #     ])
    # print(a)
    T = DecisionTree(data)
    # print(T.calc_gain(a, 0))
    # print(T.calc_gain(a, 1))
    # print(T.calc_gain(a, 2))
    # print(T.find_max_gain(a, np.array([1, 1, 0])))
    T.start_learn(data)
    T.show_tree()