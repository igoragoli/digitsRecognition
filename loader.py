"""
loader.py

Helper program to load mnist data.
"""

import numpy as np
import mnist

def load_data():
    """
    Returns the tuple "(train_set, test_set)".

    "train_set" contains 60000 training examples (x, y), and "test_set"
    contains 10000 training examples (x, y).
    "x" has shape (n^{[0]}, m) and contains n^{[0]} input features for 
    m training examples. "y" has shape (n^{[L]}, m), and contains n^{[L]}
    outputs for m training examples.
    """

    train_images = mnist.train_images()
    train_labels = np.array([label_vector(i) for i in mnist.train_labels()])
    train_labels = np.transpose(train_labels, (1, 2, 0))
    train_labels = train_labels.reshape(train_labels.shape[0] * train_labels.shape[1], train_labels.shape[2])
    
    test_images = mnist.test_images()
    test_labels = np.array([label_vector(i) for i in mnist.test_labels()])
    test_labels = np.transpose(test_labels, (1, 2, 0))
    test_labels = test_labels.reshape(test_labels.shape[0] * test_labels.shape[1], test_labels.shape[2])
    
    x_train = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]).T 
    y_train = train_labels

    x_test = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]).T
    y_test = test_labels

    train_set = (x_train, y_train)
    test_set = (x_test, y_test)

    return (train_set, test_set)

def label_vector(num):
    """
    Converts numerical labels to an (10, 1) vector.
    
    The position corresponding to num is "1", and the other
    positions are "0". This converts the numerical label into
    a desired output from the neural network.
    """
    vector = np.zeros((10, 1))
    vector[num] = 1
    return vector


#a, b = load_data()
#y = a[1]
#print(type(y.shape[0]))