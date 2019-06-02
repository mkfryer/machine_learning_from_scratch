"""
Decision Tree
"""
import numpy as np
import pandas as pd

class Branch:

    def __init__(self, category):
        self.category = category
        self.c_node = None

    def add_connection(self, node):
        self.c_node = node

class Node:
    def __init__(self, data, attribute):
        self.data = data
        self.branches = []
        self.attribute = attribute

    def connect_branch(self, branch):
        self.branches.append(branch)

    def get_class(self, s = ""):
        s = s + " node:" + self.attribute
        for branch in self.branches:
            if type(branch.c_node) == Node:
                branch.c_node.get_class(s + " branch:" + branch.category)
            elif type(branch.c_node) == LeafNode:
                print(s + " branch:" + branch.category +  " label:" + branch.c_node.label)
                
        print("exausted branches")

class LeafNode(Node):
    def __init__(self, data, label):
        super().__init__(data, None)
        self.label = label


class DecisionTree:
    def __init__(self, data, attributes):
        """
        """
        self.data = data
        self.m, self.n = data.shape
        self.root = None
        self.attributes = attributes
    
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
        entr_global = self.calc_entropy(data[:, -1])
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
        splits = {}
        for attr in attributes:
            idx_mask, = np.where(data[:, i] == attr)
            splits[attr] = data[idx_mask, :]
        return splits
        
    def start_learn(self, data):
        #attributes to split on. Can split if attr_indx = 1, 0 means you cant use it
        attrs_idx_states = np.ones(self.n)
        #dont split on category
        attrs_idx_states[-1] = 0
        #index corresponding to best gain
        bst_attr_idx = self.find_max_gain(data, attrs_idx_states)
        self.root = Node(data, self.attributes[bst_attr_idx])

        data_attr_splits = self.split_data(data, bst_attr_idx)
        # avail_attrs_idxs = np.delete(avail_attrs_idxs, [bst_attr_idx], None)
        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        for key in data_attr_splits.keys():
            data_split = data_attr_splits[key]
            branch = Branch(key)
            self.root.connect_branch(branch)
            self.learn(data_split, branch, attrs_idx_states.copy())

    def learn(self, data, branch, attrs_idx_states):
        #base case - stop learning when data is pure
        if len(np.unique(data[:, -1])) == 1:
            #create leaf node and get label
            child = LeafNode(data, data[0, -1])
            branch.add_connection(child)
            return

        bst_attr_idx = self.find_max_gain(data, attrs_idx_states)
        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        child = Node(data, self.attributes[bst_attr_idx])
        branch.add_connection(child)

        data_attr_splits = self.split_data(data, bst_attr_idx)
        for key in data_attr_splits.keys():
            data_split = data_attr_splits[key]
            branch = Branch(key)
            child.connect_branch(branch)
            self.learn(data_split, branch, attrs_idx_states.copy())

    def show_tree(self):
        """ """
        self.root.get_class()

if __name__ == "__main__":

    f =  np.array([
            ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
            ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
            ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
            ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "strong"],
            ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes" , "yes", "yes", "no"],
        ]).T
    attributes = ["outlook", "temperature", "humidity" ,"wind", "playtennis", "no", "yes"]
    T = DecisionTree(f, attributes)
    T.start_learn(f)
    T.show_tree()

    # f = np.chararray([a, b, c, d, e])

    # print(f)

    # category_encoding = np.zeros((len(f[0]), len(f)))
    # for i, x in enumerate(f):
    #     category_encoding[:, i] = pd.Series(x, dtype="category").cat.codes.values
    # attribute_encoding = pd.Series(attributes, dtype="category").cat.codes.values

    # data = np.zeros((len(f[0]), len(f)))
    # for i in len(f):
    #     data[:, i] = category_encoding[i]
    # print(category_encoding)
    # keys = ["sunny", "rainy", "overcast", "hot", "mild", "cool", "high", "normal", "weak", "strong", "no", "yes"]
    # d_k = {key:x for x,key in enumerate(keys)}
    # print(keys)
    # cols = ["outlook", "temperature", "humidity" ,"wind", "playtennis"]
    # k_c = {key:x for x,key in enumerate(cols)}
    # data = np.zeros((14, 5))
    
    # terms = ["sunny", "rainy", "overcast", "hot", "mild", "cool", "high", "normal", "weak", "strong", "no", "yes", "outlook", "temperature", "humidity" ,"wind", "playtennis"]
    # encoding = {key:x for x,key in enumerate(cols)}

    # for j, x in enumerate(f):
    #     for i, y in enumerate(x):
    #         data[i, j] = category_encoding[j, i]

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
    # print(T.calc_gain(a, 0))
    # print(T.calc_gain(a, 1))
    # print(T.calc_gain(a, 2))
    # print(T.find_max_gain(a, np.array([1, 1, 0])))
