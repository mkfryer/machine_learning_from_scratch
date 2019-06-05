"""
Decision Tree
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.animation
from matplotlib import pyplot as plt
import math 

class Branch:

    def __init__(self, category_idx, popularity):
        self.category_idx = category_idx
        self.c_node = None
        self.popularity = popularity

    def add_connection(self, node):
        self.c_node = node

class Node:
    def __init__(self, data, attribute, attribute_idx, feature_labels):
        self.data = data
        self.branches = {}
        self.attribute = attribute
        self.attribute_idx = attribute_idx
        self.most_common_branch = None
        self.feature_labels = feature_labels

    def connect_branch(self, branch, category_idx):
        if type(self.most_common_branch) != Branch or self.most_common_branch.popularity > branch.popularity:
            self.most_common_branch = branch
        self.branches[category_idx] = branch

    def predict(self, x):
        category_idx = x[self.attribute_idx]
        if category_idx in self.branches.keys():
            return self.branches[category_idx].c_node.predict(x)
        else:
            return self.most_common_branch.c_node.predict(x)

    def get_class(self, s = ""):
        s = s + " node:" + self.attribute
        for key in self.branches.keys():
            if type(self.branches[key].c_node) == Node:
                self.branches[key].c_node.get_class(s + " branch:" + key)
            elif type(self.branches[key].c_node) == LeafNode:
                print(s + " branch:" + key +  " label:" + self.branches[key].c_node.label)
                
        print("exausted branches")

    def count_children(self):
        count = len(self.branches.keys())
        for key in self.branches.keys():
            if type(self.branches[key].c_node) == Node:
                count += self.branches[key].c_node.count_children()
        return count

    def get_depth(self):
        depths = [0]
        for key in self.branches.keys():
            if type(self.branches[key].c_node) == Node:
                depths.append(self.branches[key].c_node.get_depth())
            elif type(self.branches[key].c_node) == LeafNode:
                depths.append(1)
        return 1 + max(depths)

    def add_edges(self, G, labels):
        for branch in self.branches.values():
            if type(branch.c_node) == Node:
                G.add_node(branch.c_node.attribute)
                G.add_edge(branch.c_node.attribute, self.attribute)

                labels[(branch.c_node.attribute, self.attribute)] = self.feature_labels[self.attribute_idx][branch.category_idx] + ":" + str(branch.popularity)
                branch.c_node.add_edges(G, labels)
            elif type(branch.c_node) == LeafNode:
                """ """
                id_n = str(self.attribute[:2]) + "-" + str(branch.c_node.label)
                G.add_node(id_n)  
                G.add_edge(id_n, self.attribute)

                labels[(id_n, self.attribute)] = self.feature_labels[self.attribute_idx][branch.category_idx] + ":" + str(branch.popularity)

        return labels

    def prune(self, get_accuracy):
        for branch in self.branches.values():
            if type(branch.c_node) == Node:
                branch.c_node.prune(get_accuracy)
        
        accuracy = get_accuracy()
        unhatched_branches = self.branches
        self.branches = {}
        new_accuracy = get_accuracy()
        if new_accuracy < accuracy:
            self.branches = unhatched_branches
    

    def eliminate_children(self):
        #todo - manage memory better...
        self.branches = {}

    def attatch_children(self, children):
        self.branches = children

class LeafNode(Node):
    def __init__(self, data, label):
        super().__init__(data, None, None, None)
        self.label = label

    def predict(self, x):
        return self.label

class DecisionTree:
    def __init__(self, train_data, test_data, validation_data, attributes, feature_labels):
        """
        """
        self.test_data = test_data
        self.train_data = train_data
        self.validation_data = validation_data
        self.m, self.n = train_data.shape
        self.root = None
        self.attributes = attributes
        self.feature_labels = feature_labels

    def prune(self):
        get_accuracy = lambda: self.predict_set(self.validation_data[:, :-1], self.validation_data[:, -1])
        self.root.prune(get_accuracy)

    def get_node_count(self):
        return 1 + self.root.count_children()

    def get_depth(self):
        return self.root.get_depth()
    
    def calc_entropy(self, targ_col_data):
        uniq = np.unique(targ_col_data)
        n = len(targ_col_data)
        m = len(uniq)
        p = np.zeros(m).astype(float)
        for i in range(m):
            z = np.where(targ_col_data == uniq[i])[0]
            p[i] = len(z)/n
        return -p @ np.log2(p)

    def calc_gain(self, data, attr_index, split_info):
        """ """
        m, n = data.shape
        entr_global = self.calc_entropy(data[:, -1])
        categories = np.unique(data[:, attr_index])
        sub_entr = np.zeros(len(categories)).astype(float)
        split_info_val = 0.0

        for i, cat in enumerate(categories):
            cat_mask = np.where(data[:, attr_index] == cat)[0]
            c = len(cat_mask)/m
            sub_entr[i] = c * self.calc_entropy(data[:, -1][cat_mask])
            split_info_val -= c * math.log(c, 2)

        gain = entr_global - np.sum(sub_entr)
        if split_info:
            gain /= split_info_val

        return gain

    def find_max_gain(self, data, attrs_idx_states, split_info):
        n = len(attrs_idx_states)
        gains = np.zeros(n).astype(float)
        active_attr_indxs, = np.where(attrs_idx_states == 1)
        for attr_idx in active_attr_indxs:
            gains[attr_idx] = self.calc_gain(data, attr_idx, split_info)
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
        
    def start_learn(self, data, split_info = False):
        #attributes to split on. Can split if attr_indx = 1, 0 means you cant use it
        attrs_idx_states = np.ones(self.n)
        #dont split on category
        attrs_idx_states[-1] = 0
        #index corresponding to best gain
        bst_attr_idx = self.find_max_gain(data, attrs_idx_states, split_info)

        self.root = Node(data, self.attributes[bst_attr_idx], bst_attr_idx, self.feature_labels)

        data_attr_splits = self.split_data(data, bst_attr_idx)
        # avail_attrs_idxs = np.delete(avail_attrs_idxs, [bst_attr_idx], None)
        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        for key in data_attr_splits.keys():
            data_split = data_attr_splits[key]
            branch = Branch(key, len(data_split[:, 0]))
            self.root.connect_branch(branch, key)
            self.learn(data_split, branch, attrs_idx_states.copy(), split_info)
        

    def learn(self, data, branch, attrs_idx_states, split_info):
        #base case - stop learning when data is pure
        if len(np.unique(data[:, -1])) == 1:
            #create leaf node and get label
            child = LeafNode(data, data[0, -1])
            branch.add_connection(child)
            return

        bst_attr_idx = self.find_max_gain(data, attrs_idx_states, split_info)
        #set state used attr_idx as used
        attrs_idx_states[bst_attr_idx] = 0
        child = Node(data, self.attributes[bst_attr_idx], bst_attr_idx, self.feature_labels)
        branch.add_connection(child)

        data_attr_splits = self.split_data(data, bst_attr_idx)
        for key in data_attr_splits.keys():
            data_split = data_attr_splits[key]
            if len(data_split[:, 0]) > 0:
                branch = Branch(key, len(data_split[:, 0]))
                child.connect_branch(branch, key)
                self.learn(data_split, branch, attrs_idx_states.copy(), split_info)


    def show_tree(self):
        """ """
        edge_labels = {}
        G = nx.Graph()
        G.add_node(self.root.attribute)
        self.root.add_edges(G, edge_labels)
        pos = nx.spring_layout(G, k=0.05, iterations=20)

        plt.figure(3, figsize=(11,11)) 
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(G, edge_labels=edge_labels, pos=pos)

        plt.show()

    def predict(self, x, y):
        label = self.root.predict(x)
        return label
        # print("predicting", x, y, label)

    def predict_set(self, data, labels):
        n = len(data[:, 0])
        num_correct = 0
        for i in range(n):
            if self.predict(data[i, :], labels[i]) == labels[i]:
                num_correct += 1
        return num_correct/n


if __name__ == "__main__":

    f =  np.array([
            ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
            ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
            ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
            ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "weak", "weak", "strong", "strong", "weak", "strong"],
            ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes" , "yes", "yes", "no"],
        ]).T
    
    attributes = ["outlook", "temperature", "humidity" ,"wind", "playtennis", "no", "yes"]
    f_labels = ["yes", "no"]
    # T = DecisionTree(f, None, None, attributes, f_labels)
    # T.start_learn(f, True)
    # print("node count", T.get_node_count())
    # print("depth", T.get_depth())
    # T.root.get_class()
    # T.show_tree()

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
