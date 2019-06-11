from toolkit.arff import Arff
from KNearestNeighbors import KNNClassifier
import numpy as np
import matplotlib.pyplot as plt
import time


def show_table(collabel, rowlabel, m, title = ""):
    fig, axs = plt.subplots(1,1)
    axs.axis('tight')
    axs.axis('off')
    axs.table(
            cellText=m,
            colLabels=collabel,
            rowLabels=rowlabel,
            loc='center'
        )
    axs.set_title(title)
    plt.show()

def prob_2(weighted_d = False):
    """ """
    k = 3
    test_arff = Arff("magic_telescope_testing_data.arff")
    train_arff = Arff("magic_telescope_training_data.arff")
    test_arff.shuffle()
    train_arff.shuffle()

    # attributes = test_arff.get_attr_names()
    test_data = np.hstack((test_arff.get_features().data, test_arff.get_labels().data))
    train_data = np.hstack((train_arff.get_features().data, train_arff.get_labels().data))
    KNNC = KNNClassifier(k, train_data, test_data)
    acc = KNNC.get_accuracy(weighted_d)

    test_arff.normalize()
    train_arff.normalize()
    n_test_data = np.hstack((test_arff.get_features().data, test_arff.get_labels().data))
    n_train_data = np.hstack((train_arff.get_features().data, train_arff.get_labels().data))
    n_KNNC = KNNClassifier(k, n_test_data, n_train_data)
    acc_n = n_KNNC.get_accuracy(weighted_d)

    # print(np.array([[acc,acc_n]]))
    print(acc,acc_n)
    # show_table(["Not Normalized"  "Normailzed"], ["Accuracy"], np.array([[acc,acc_n]]), title = "Normalized vs Non-normalized, k=3")

    K = [1, 3, 5, 7, 9, 11, 13, 15]
    A = []
    for k_hat in K:
        # n_test_data = np.hstack((test_arff.get_features().data, test_arff.get_labels().data))
        # n_train_data = np.hstack((train_arff.get_features().data, train_arff.get_labels().data))
        n_KNNC = KNNClassifier(k_hat, n_train_data, n_test_data)
        A.append(n_KNNC.get_accuracy(weighted_d))

    plt.plot(K, A, label="")
    t = "KNN Accuracy Telesc. "
    if weighted_d:
        t += "(weighted-d)"
    plt.title(t)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.show()
    
def prob_3(weighted_d = False):
    test_arff = Arff("housing_testing_data.arff")
    train_arff = Arff("housing_training_data.arff")
    test_arff.shuffle()
    train_arff.shuffle()
    test_arff.normalize()
    train_arff.normalize()

    K = [1, 3, 5, 7, 9, 11, 13, 15]
    A = []
    for k_hat in K:
        test_data = np.hstack((test_arff.get_features().data, test_arff.get_labels().data))
        train_data = np.hstack((train_arff.get_features().data, train_arff.get_labels().data))
        KNNC = KNNClassifier(k_hat, train_data, test_data)
        A.append(KNNC.get_accuracy_regress(weighted_d))
    
    plt.plot(K, A, label="")
    t = "KNN Regression M.S.E Housing"
    if weighted_d:
        t += "(weighted-d)"
    weighted_d
    plt.title(t)
    plt.xlabel("K")
    plt.ylabel("M.S.E")
    # plt.legend()
    plt.show()

def prob_4():
    prob_2(True)
    # prob_3(True)

def prob_5():
    cont_mask = [1, 2, 7, 10, 13, 14, 16]
    cate_mask = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]

    arff = Arff("credit_approval_data.arff")
    arff.shuffle()
    arff.normalize()

    n = len(arff.get_labels().data)
    t = int(n * .7)
    train_data = arff.create_subset_arff(row_idx=slice(0, t, 1))
    test_data = arff.create_subset_arff(row_idx=slice(t, n, 1))
    test_data = np.hstack((test_data.get_features().data, test_data.get_labels().data))
    train_data = np.hstack((train_data.get_features().data, train_data.get_labels().data))
    #b,30.83,0,u,g,w,v,1.25,t,t,01,f,g,00202,0,+
    dist_matrix = np.ones((16, 16))
    np.fill_diagonal(dist_matrix, 0)
    KNNC = KNNClassifier(8, train_data, test_data)
    print(KNNC.get_accuracy_mixed(cate_mask, cont_mask, dist_matrix))

def prob_6():
    """ """
    k = 3
    test_arff = Arff("magic_telescope_testing_data.arff")
    train_arff = Arff("magic_telescope_training_data.arff")
    test_arff.shuffle()
    train_arff.shuffle()
    test_arff.normalize()
    train_arff.normalize()

    K = [1, 3, 5]
    T = []
    A = []
    T_KSM = []
    A_KSM = []
    for k_hat in K:
        test_data = np.hstack((test_arff.get_features().data, test_arff.get_labels().data))
        train_data = np.hstack((train_arff.get_features().data, train_arff.get_labels().data))
        KNNC = KNNClassifier(k_hat, train_data, test_data)

        t = time.time()
        A.append(KNNC.get_accuracy())
        T.append(time.time() - t)
        KNNC.induce_KSM()

        t = time.time()
        A_KSM.append(KNNC.get_accuracy())
        T_KSM.append(time.time() - t)

    ax = plt.axes(projection='3d')
    ax.plot(K, A, T, label="No-KSM")
    ax.plot(K, A_KSM, T_KSM, label="KSM")

    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_zlabel('Time')

    t = "KNN Accuracy w/ IKSM"
    plt.title(t)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    """ """
    # prob_2()
    # prob_3()
    # prob_4()
    prob_5()
    # prob_6()