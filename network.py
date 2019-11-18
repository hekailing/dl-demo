#!/usr/bin/python
import numpy as np
import random

def Sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def Sigmoid_prime(z):
    sig = Sigmoid(z)
    return sig * (1 - sig)


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        给定输入"a", 返回10维的输出
        """
        for b, w in zip(self.biases, self.weights):
            a = Sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                ## 使用梯度下降更新参数
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)}, {n_test}")
            else:
                print(f"Epoch {j}: complete")

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 构造和self.biases一样大小的列表，用来存放累加的梯度
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 对训练数据(x,y)计算Loss相对于所有参数的偏导数
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 累加梯度到nabla_b和nabla_w中
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        # 使用梯度和步长eta更新参数weights和biases

    def cost_derivative(self, x, y):
        x = x.transpose()
        for e in x:
            e -= y
        return x.transpose()

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播计算
        activation = x
        activations = [x] # 用来存储激活的列表
        zs = [] # 用来存储z的列表
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Sigmoid(z)
            activations.append(activation)

        # 反向传播计算
        delta = self.cost_derivative(activations[-1], y) * Sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        print(nabla_b[-1], nabla_w[-1])
        return (nabla_b, nabla_w)
