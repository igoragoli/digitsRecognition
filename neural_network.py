"""
neural_network.py

Implements a L-layer neural network.
"""

import numpy as np 

class NeuralNetwork():
    def __init__(self, layers_dims):
        
        self.layers_dims = layers_dims
        self.parameters = {}
        self.L = len(self.layers_dims)

        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * np.sqrt(2 / self.layers_dims[l])
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def forward_propagation(self, X):
        caches = []
        A = X 
        # ReLU
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['W' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.g(Z, "relu")
            caches = caches.append((A_prev, W, b, Z))
        
        # Sigmoid
        A_prev = A
        W = self.parameters['W' + str(self.L)]
        b = self.parameters['b' + str(self.L)]
        ZL = np.dot(W, A_prev) + b
        AL = self.g(ZL, "sigmoid")
        caches = caches.append((A, W, b, ZL))

        return AL, caches

    def backward_propagation_unit(self, l, dA, cache):
        W, A_prev, Z = cache

        dZ = dA * self.dg(l, Z)
        dW = (1 / self.m) * np.dot(dZ, A_prev.T)
        db = (1 / self.m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def g(self, l, a):
        if l == self.L: # Sigmoid
            r = 1 / (1 + np.exp(-a))
        else: # ReLU
            if a >= 0:
                r = a
            else:
                r = 0

        return r

    def dg(self, l, a, *args):
        if l == self.L: # Sigmoid
            y = args
            r = - (y / a) + (1 - y) / (1 - a)
        else: # Relu
            if a >= 0:
                r = 1
            else:
                r = 0
        
        return r