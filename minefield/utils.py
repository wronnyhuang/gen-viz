"""A set of utility functions used throughout this repo.

"""
import os
import keras
import math
import matplotlib
matplotlib.use('agg')
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf


from keras import initializers, regularizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Flatten
from keras.backend import binary_crossentropy
from tensorflow.python.ops.gradients_impl import _hessian_vector_product


def create_untraining_set(test_data, test_label):
    """Split the test data in half: into test and untraining data
    Note that the test data is not shuffled, so the first half of the data is
    class zero and the second half is class one. Below we choose to sample
    randomly from the array.

    Args:
        test_data (np.array): The test data.
        test_label (np.array): The labels of the test data.

    Returns: (Dict)

    """
    #
    size_untraining = math.floor(0.5 * test_data.shape[0])
    idx = np.random.choice(np.arange(test_data.shape[0]),
                           size_untraining,
                           replace=False)
    untraining_data = test_data[idx]
    untraining_label = test_label[idx]
    test_idx = np.delete(np.arange(test_data.shape[0]), idx)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]
    # unique, counts = np.unique(train_label, return_counts=True)
    # print(dict(zip(unique, counts)))

    # Modify labels for the untraining data to have the loss function
    # treat it differently. Basically switch the labels on the untraining set.

    untraining_label_true = untraining_label
    untraining_label = np.array([1 if j == 0 else 0
                                 for j in untraining_label.tolist()])

    return {"untraining_data": untraining_data,
            "untraining_label": untraining_label,
            "untraining_label_true": untraining_label_true,
            "test_data": test_data,
            "test_label": test_label}


def fully_connected(n_hidden_layers=3,
                    n_neurons_per_layer=10,
                    use_regularizers=True):
    """
    Creates a fully connected neural net.
    Args:
        n_hidden_layers (int): Number of hidden layers.
        n_neurons_per_layer (int): Number of neurons per hidden layer.
        use_regularizers (bool): Whether to include kernel regularizers
            (i.e. weight decay).

    Returns: model (keras.model.Sequential) The generated model.

    """
    if use_regularizers:
        regularizer = regularizers.l2(0.001)
    else:
        regularizer = None
    model = Sequential()
    model.add(Dense(n_neurons_per_layer,
                    activation='relu',
                    input_dim=2,
                    kernel_regularizer=regularizer,
                    kernel_initializer=initializers.glorot_normal(seed=None)))

    for _ in range(n_hidden_layers - 2):
        model.add(Dense(n_neurons_per_layer,
                        activation='relu',
                        kernel_regularizer=regularizer,
                        kernel_initializer=initializers.glorot_normal(
                            seed=None)))

    model.add(Dense(1,
                    activation='sigmoid',
                    kernel_regularizer=regularizer,
                    kernel_initializer=initializers.glorot_normal(seed=None)))

    return model


def generate_spiral_data(n_points=1000,
                         n_cycles=5,
                         noise_std_dev=.1,
                         random_seed=0,
                         train_ratio=.8):

    """
    This function generates a 2 class (red=1 and blue=0) spiral dataset
    with n_points in each class. train_ratio is the percentage of the
    dataset that would be reserved for training, the rest is used for eval. 
    The function returns the train_data, test_data, train_label_data, and
    test_label_data. 
    """
    # Set random seed and generate noise vectors for each class
    np.random.seed(random_seed)
    red_noise = np.random.normal(0, noise_std_dev, [2, n_points])
    blue_noise = np.random.normal(0, noise_std_dev, [2, n_points])

    # Generate Data
    theta_max = n_cycles * (2 * math.pi)
    step_size = theta_max/n_points
    red_data = [[5 * math.sqrt(t * step_size) *
                 math.cos(t * step_size) + red_noise[0][t],
                 5 * math.sqrt(t * step_size) *
                 math.sin(t * step_size) + red_noise[1][t],
                 1]
                for t in range(n_points)]

    blue_data = [[-5 * math.sqrt(t * step_size) *
                  math.cos(t * step_size) + blue_noise[0][t],
                  -5 * math.sqrt(t * step_size) *
                  math.sin(t * step_size) + blue_noise[1][t],
                  0]
                 for t in range(n_points)]

    # Split the data into train and eval
    test_data = red_data + blue_data
    n_train_pts = math.ceil(2 * n_points * train_ratio)
    train_data = random.sample(test_data, k=n_train_pts)
    for i in train_data:
        test_data.remove(i)
    x, y, train_label = zip(*train_data)
    train_data = np.array(list(zip(x, y)))
    x, y, test_label = zip(*test_data)
    test_data = np.array(list(zip(x, y)))

    return train_data, test_data, np.array(train_label), np.array(test_label)


def list2dotprod(listoftensors1, listoftensors2):
    """Computes the dot product of two lists of tensors (such as those returned
    when you call tf.gradients) as if each list were one concatenated tensor.

    Args:
        listoftensors1:
        listoftensors2:

    Returns:

    """
    return tf.add_n([tf.reduce_sum(tf.multiply(a, b))
                     for a, b in zip(listoftensors1, listoftensors2)])


def list2norm(list_of_tensors):
    """Computes the 2-norm of a list of tensors (such as those returned when you
    call tf.gradients) AS IF list were one concatenated tensor

    Args:
        listOfTensors:

    Returns:

    """
    return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(a))
                             for a in list_of_tensors]))


def normalize(x, order=2):
    """
    Normalizes a matrix x. The norm used depends on the order parameter.
    Args:
        x (np.ndarray): Matrix to be normalized.
        order (int): Order of the norm to be used. Refer to numpy docs for more
            info.

    Returns: (np.ndarray)

    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, -1))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, -1)


def plot_data(train_data,
              train_label,
              filename: str,
              test_data=None,
              test_label=None):
    """
    This function plots the input data and saves the plot as a png file.
    train_data is plotted in blue (label=0) and red (label=1).
    The test_data is an optional argument and is plotted in cyan (label=0)
    and orange (label=1).
    Args:
        train_data (np.ndarray): The training data to be plotted in red/blue.
        train_label (np.ndarray): The labels for the training data. Must be
            in the same order as the corresponding training points.
        filename (str): The name under which the plot will be saved to disk.
        test_data (np.ndarray): The test data to be plotted in orange/cyan.
        test_label (np.ndarray): The labels for the test data. Must be
            in the same order as the corresponding test points.

    Returns:

    """
    plt.figure()
    if os.path.isfile(os.path.join('./', filename)):
        os.remove(filename)
    train_label = np.squeeze(train_label)
    # print(f'The number of train labels {train_label.shape}')
    # print(train_label.min())
    # print(f'The number of red {train_label[train_label==1].shape}')
    blue = train_data[train_label == 0]
    red = train_data[train_label == 1]
    plt.scatter(red[:, 0], red[:, 1], c='red', s=0.1)
    plt.scatter(blue[:, 0], blue[:, 1], c='blue', s=0.1)

    if test_data is not None:
        test_label = np.squeeze(test_label)
        cyan = test_data[test_label == 0]
        orange = test_data[test_label == 1]

        plt.scatter(cyan[:, 0], cyan[:, 1], c='cyan', s=0.1)
        plt.scatter(orange[:, 0], orange[:, 1], c='orange', s=0.1)
    plt.savefig(fname=filename)
    pass


def plot_decision_boundary(model,
                           filename='decision_boundary',
                           npoints=500j):
    """
    Generates a mesh grid, and evaluates the model on the mesh,
    then plots the decision boundary. The plot is then saved to disk.
    Args:
        model (keras.models.Sequential): The model used to plot the decision
            boundary.
        filename (str): The name under which the plot will be saved to disk.
        npoints : Number of points on each axis of the mesh. Must be a
            complex number (i.e. end in "j" as the default value shows).

    Returns:

    """
    xmin, xmax, ymin, ymax = [-25, 25, -25, 25]
    X, Y = np.mgrid[xmin:xmax:npoints, ymin:ymax:npoints]
    mesh = np.squeeze(np.dstack((X.ravel(), Y.ravel())))
    predictions = model.predict_classes(mesh, batch_size=1000)
    plot_data(mesh, predictions, filename=filename)
    pass
