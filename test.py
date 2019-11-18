#!/usr/bin/python
import os
import mnist_loader
import network


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(path)
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3, test_data=test_data)
