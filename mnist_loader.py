#!/usr/bin/python
import os
import io
import struct
import numpy as np


def load_data(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    # import matplotlib.pyplot as plt
    # for img, lab in zip(images[:10], labels[:10]):
    #     plt.imshow(img.reshape(rows, cols))
    #     plt.title(lab)
    #     plt.show()

    return zip(images, labels)


def load_data_wrapper(path):
    train_data = load_data(path, kind='train')
    test_data = load_data(path, kind='t10k')
    train = []
    valid = []
    test = []
    train_data = list(train_data)
    for img, lab in train_data[:50000]:
        onehot = [0] * 10
        onehot[lab] = 1
        train.append((img.reshape(784, 1), onehot))
    for img, lab in train_data[50000:]:
        valid.append((img.reshape(784, 1), lab))
    for img, lab in test_data:
        test.append((img.reshape(784, 1), lab))
    return train, valid, test
