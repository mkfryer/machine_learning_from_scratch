import cv2
import os, sys
from os import listdir
from os.path import isfile, join
from Perceptron import Perceptron
import numpy as np
from sklearn.utils import shuffle


def resize_image(filename, output_dir=""):
    W = 35
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return
    resized_im = cv2.resize(im, (W,W))
    normed_im = cv2.normalize(resized_im, None, 0, 100, cv2.NORM_MINMAX)
    # cv2.imshow("Show by CV2", normed_im)
    cv2.waitKey(0)
    cv2.imwrite(output_dir, normed_im)

def save_images_as_npy():
    # files = [f for f in listdir("./images") if isfile(join("./images", f))]
    # for file in files:
    #     resize_image("./images/" + file, "./images_processed/" + file)


    path = "./images_processed/"
    k_pic_paths = [join(path + "k", f) for f in listdir(path + "k") if isfile(join(path + "k", f))]
    i_pics_paths = [join(path + "isa", f) for f in listdir(path + "isa") if isfile(join(path + "isa", f))]
    c_pics_paths = [join(path + "c", f) for f in listdir(path + "c") if isfile(join(path + "c", f))]


    dataset = np.zeros((45, 35*35 + 1))
    i = 0

    for path in k_pic_paths:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        dataset[i, :-1] = im.flatten()
        dataset[i,-1] = 1
        i += 1

    for path in i_pics_paths:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        dataset[i, :-1] = im.flatten()
        dataset[i,-1] = 0
        i += 1

    for path in c_pics_paths:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        dataset[i, :-1] = im.flatten()
        dataset[i,-1] = 2
        i += 1

    dataset.tofile("image_vecs.npy")
    np.save("image_vecs.npy", dataset)