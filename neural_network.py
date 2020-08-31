"""
neural_network.py

Implements a L-layer neural network.
"""

import numpy as np 

class NeuralNetwork():
    def __init__(self, layers_dims):
        
        self.layers_dims = layers_dims
        self.parameters = {}
        L = len(self.layers_dims)

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * np.sqrt(2 / self.layers_dims[l])
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def forward_propagation(self, l, A_prev):
        W = self.parameters['W' + str(l)]
        b = self.parameters['W' + str(l)]

        Z = np.dot(W, A_prev) + b 
        A = g(l, Z)

        return A, Z