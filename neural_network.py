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

        # Hidden layers (ReLU)
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['W' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.g(Z, "relu")
            caches = caches.append((A_prev, W, b, Z))
        
        # Output layer (Sigmoid)
        A_prev = A
        WL = self.parameters['W' + str(self.L)]
        bL = self.parameters['b' + str(self.L)]
        ZL = np.dot(W, A_prev) + bL
        AL = self.g(ZL, "sigmoid")
        caches = caches.append((A_prev, WL, bL, ZL))

        return AL, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        dAL = AL - Y

        # Output Layer
        A_prev, WL, _, ZL = caches[self.L - 1] 
        dZL = dAL * self.dg(ZL, "sigmoid")
        grads["dW" + str(self.L)]  = (1 / self.m) * np.dot(dZL, A_prev.T)
        grads["db" + str(self.L)] = (1 / self.m) * np.sum(dZL, axis=1, keepdims=True)
        grads["dA" + str(self.L - 1)] = np.dot(WL.T, dZL)
    
        # Hidden Layers
        for l in reversed(range(self.L - 1)):
            A_prev, W, _, Z = caches[l]
            dZ = grads["dA" + str(l)] * self.dg(Z, "relu")
            grads["dW" + str(l)]  = (1 / self.m) * np.dot(dZ, A_prev.T)
            grads["db" + str(l)]  = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
            grads["dA" + str(l - 1)] = np.dot(W.T, dZ)

        return grads

    def g(self, z, activation):
        if activation == "sigmoid":
            r = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            if z >= 0:
                r = z
            else:
                r = 0
        else:
            r = 0
        return r

    def dg(self, z, activation):
        if activation == "sigmoid":
            sig = self.g(z, "sigmoid")
            r = sig * (1 - sig)
        elif activation == "relu":
            if z >= 0:
                r = 1
            else:
                r = 0
        
        return r