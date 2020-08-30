"""
loader.py

Helper program to load mnist data.
"""

import numpy as np
import mnist

def load_data():
    """
    Returns the tuple "(train_set, dev_set, test_set)".
    "train_set" contains 50000 training examples (x, y). "dev_set" 
    and "test_set" contain 10000 examples, each.
    """
    # Separate original training set into training set and dev set.
    train_images_orig = mnist.train_images()
    train_labels_orig = np.array([label_vector(i) for i in mnist.train_labels()])
    train_labels_orig = np.transpose(train_labels_orig, (1, 2, 0))
    train_images = train_images_orig[0:50000, :, :]
    train_labels = train_labels_orig[:, :, 0:50000]
    dev_images = train_images_orig[50000:60000, :, :]
    dev_labels = train_labels_orig[:, :, 50000:60000]
    
    test_images = mnist.test_images()
    test_labels = np.array([label_vector(i) for i in mnist.test_labels()])
    test_labels = np.transpose(test_labels, (1, 2, 0))
    
    x_train = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]).T 
    y_train = train_labels

    x_dev = dev_images.reshape(dev_images.shape[0], dev_images.shape[1] * dev_images.shape[2]).T
    y_dev = dev_labels

    x_test = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]).T
    y_test = test_labels

    train_set = (x_train, y_train)
    dev_set = (x_dev, y_dev)
    test_set = (x_test, y_test)

    return (train_set, dev_set, test_set)

def label_vector(num):
    """
    Converts numerical labels to an (10, 1) vector in which
    the position corresponding to num is "1", and the other
    positions are "0". This converts the numerical label into
    a desired output from the neural network.
    """
    vector = np.zeros((10, 1))
    vector[num] = 1
    return vector