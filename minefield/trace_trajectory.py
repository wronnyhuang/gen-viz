"""This script supplements train_good_model.py. Given the saved weights at
each epoch of training this script can perform 2 tasks: trace and process.
Tracing involves intializing a network at different epochs and searching for a
nearby bad minimum. The weights for each bad minimum can also be saved to disk.
To keep things simple, most of the information relevant from a math/optimization
perspective is set while running train_good_model.py and is not affected here.
Tracing only relies on a few command line arguments explained below. Notably,
the argument seed_value specifies which folder to use for the "good_weights".
Some parameters are hard-coded for simplicity:
    1) The number of epochs and the batch size for each "untraining run"
    2) To generate the untraining data the test data is split in half

Processing takes place after tracing and consists of a series of tasks to
evaluate the bad minima and generate several plots.
The tasks performed include:
    1) Compute distances between good weights and corresponding bad weights.
    2) Perform PCA and generate tSNE plots of weights
"""
import argparse
import os
import keras
import math
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf

from keras.backend import binary_crossentropy
from matplotlib.ticker import NullFormatter
from time import time
from typing import Dict
from utils import create_untraining_set, fully_connected, normalize

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser()
parser.add_argument("--seed_value",
                    default=0,
                    type=int,
                    help="The value of the random seed used to generate "
                         "the good net. In the context of this script this "
                         "parameter is only used to locate the saved weights "
                         "that will be used. ")

parser.add_argument("--save_weights",
                    default=False,
                    type=bool,
                    help="Whether or not to save the weights for each bad min."
                         "Has no effect if tracing is set to false.")

parser.add_argument("--epoch_start",
                    default=1,
                    type=int,
                    help="The epoch at which to start tracing or processing.")

parser.add_argument("--epoch_end",
                    default=2000,
                    type=int,
                    help="The epoch at which to end tracing or processing.")


parser.add_argument("--trace",
                    default=False,
                    type=bool,
                    help="Load good model at each checkpoint and look for bad "
                         "minima at each epoch starting from epoch_start.")

parser.add_argument("--process",
                    default=False,
                    type=bool,
                    help="Load all checkpoints, compute distances from good "
                         "minima to bad minima initialized at that checkpoint "
                         "and create t-SNE plot.")

parser.add_argument("--epoch_modulo",
                    default=1,
                    type=int,
                    help="Use if you do not want to find a bad minimum "
                         "starting at every checkpoint file. If set to N, "
                         "then starting at epoch_start the algorithm will "
                         "only trace the trajectory by initializing at every "
                         "Nth checkpoint (i.e. skipping N checkpoint files "
                         "between runs). Works for both trace and process.")

parser.add_argument("--tsne_seed",
                    default=None,
                    type=int,
                    help="The random seed to be used by TSNE.")

# Parse command line arguments
args = parser.parse_args()
seed_value = args.seed_value
save_weights = args.save_weights
epoch_start = args.epoch_start
epoch_end = args.epoch_end
trace = args.trace
process = args.process
epoch_modulo = args.epoch_modulo
tsne_seed = args.tsne_seed

if tsne_seed is None:
    tsne_seed = np.random.randint(500)


def organize(model_params: Dict,
             good_weights_filepath: str,
             bad_weights_filepath: str,
             data):
    """ Organizes a model's weights to be used for plotting later. Creates a
    model using utils.fully_connected() and the parameters in model_params
    then loads two sets of model weights specified by their filepaths to the
    hdf5 files.
    This function also computes distances between the good_weights and the
    bad_weights. The distance is the sum of distances d_i per layer divided
    by the number of layers. For each layer i, d_i is distances between the
    normalized good_weights and the normalized bad_weights of that layer. The
    normalized good_weights are computed using utils.normalize(), while the
    normalized bad_weights normalized by the l_2 norm of the good_weights for
    that layer.
    Args:
        model_params (Dict): Contains model params as stored by
            train_good_model.py. In particular w params are relevant:
                1) n_hidden_layers
                2) n_neurons_per_layer
        good_weights_filepath (str): Path to the file containing the
            weights of the good model.
        bad_weights_filepath (str): Path to the file containing the
            weights of the bad model.
        data (Dict): Dict containing the data.

    Returns:
            distance (float): Average distance between normalized layers of
                good and bad model as explained above.
            flat_good_weights(np.ndarray): A 1-D vector containing the weights
                of good model in increasing order of layers flattened in
                "row major" order.
            flat_bad_weights(np.ndarray): A 1-D vector containing the weights
                of bad model in increasing order of layers flattened in
                "row major" order.
            flat_good_biases(np.ndarray): A 1-D vector containing the biases
                of good model in increasing order of layers.
            flat_bad_biases(np.ndarray): A 1-D vector containing the biases
                of bad model in increasing order of layers.
            test_acc (float): The accuracy of the model on the test data.

    """
    train_data = data['train_data']
    train_label = data['train_label']
    test_data = data['test_data']
    test_label = data['test_label']

    good_model = fully_connected(model_params["n_hidden_layers"],
                                 model_params["n_neurons_per_layer"],
                                 use_regularizers=False)
    good_model.load_weights(good_weights_filepath)

    bad_model = fully_connected(model_params["n_hidden_layers"],
                                model_params["n_neurons_per_layer"],
                                use_regularizers=False)

    bad_model.load_weights(bad_weights_filepath)
    optimizer = keras.optimizers.RMSprop(lr=0.001,
                                         decay=0)
    bad_model.compile(optimizer=optimizer,
                      loss=binary_crossentropy,
                      metrics=['binary_accuracy'])

    test_acc = bad_model.evaluate(test_data, test_label, batch_size=32,
                                  verbose=0)[1]
    train_acc = bad_model.evaluate(train_data, train_label, batch_size=32,
                                   verbose=0)[1]

    distance = 0
    num_layers = len(good_model.layers)
    flat_good_weights = []
    flat_bad_weights = []
    flat_good_biases = []
    flat_bad_biases = []
    for j in range(num_layers):
        good_weights = good_model.layers[j].get_weights()[0]
        good_biases = good_model.layers[j].get_weights()[1]
        good_weights_normalized = normalize(good_weights)
        good_biases_normalized = normalize(good_biases)
        l2_weights = np.atleast_1d(np.linalg.norm(good_weights, ord=2, axis=-1))
        l2_biases = np.atleast_1d(np.linalg.norm(good_biases, ord=2, axis=-1))
        l2_weights[l2_weights == 0] = 1
        l2_biases[l2_biases == 0] = 1

        bad_weights = bad_model.layers[j].get_weights()[0]
        bad_weights_normalized = bad_weights / np.expand_dims(l2_weights, -1)

        bad_biases = bad_model.layers[j].get_weights()[1]
        bad_biases_normalized = np.expand_dims(l2_biases, -1)

        distance += (np.linalg.norm(good_weights_normalized -
                                    bad_weights_normalized, ord=2) +
                     np.linalg.norm(good_biases_normalized -
                                    bad_biases_normalized, ord=2)) / 2
        flat_good_weights = np.append(flat_good_weights,
                                      np.ndarray.flatten(good_weights))
        flat_bad_weights = np.append(flat_bad_weights,
                                     np.ndarray.flatten(bad_weights))
        flat_good_biases = np.append(flat_good_biases,
                                     np.ndarray.flatten(good_biases))
        flat_bad_biases = np.append(flat_bad_weights,
                                    np.ndarray.flatten(bad_biases))

    # Clear the Model and all the weights from GPU (otherwise Keras keeps the
    # session and things become incredibly slow)
    keras.backend.clear_session()
    distance /= num_layers
    return (distance, flat_good_weights, flat_bad_weights, flat_good_biases,
            flat_bad_biases, test_acc, train_acc)


def find_bad_min(model_params: Dict,
                 weights_filepath: str,
                 data: Dict,
                 save_weights: bool = False,
                 epoch_number: int = None,
                 seed_value: int = None):
    """ Initialize model specified by model_params at weights specified by
    weights_filepath and train using the data. The data Dict is assumed to
    contain untraining data, i.e. training data for which the labels were
    flipped.
    The bad weights are stored as
        f"saved_models/seed_value/bad_model/bad_model_{epoch_number}.hdf5"

    The untraining parameters (n_epochs, batch_size, etc...) are hard-coded.

    Args:
        model_params (Dict): Contains model params as stored by
            train_good_model.py. In particular w params are relevant:
                1) n_hidden_layers
                2) n_neurons_per_layer
        weights_filepath:
        data (Dict): Dict containing the data.
        save_weights(bool): If True, the weights at the conclusion of the
            untraining are saved to disk.
        epoch_number (int): Should match the the epoch number at the end of
            weights_filepath. Used to name the weights to be saved.
        seed_value (int):

    Returns: score (Dict): A dict mapping string ("train", "untrain", and
                "test") to float. The strings correspond to datasets and the
                floats to the untrained model accuracy on that data.
    """

    # Unpack the data dict.
    train_data = data['train_data']
    train_label = data['train_label']
    untraining_data = data['untraining_data']
    untraining_label = data['untraining_label']
    test_data = data['test_data']
    test_label = data['test_label']
    untraining_label_true = data['untraining_label_true']

    # Generate the model and load weights.
    model = fully_connected(model_params["n_hidden_layers"],
                            model_params["n_neurons_per_layer"],
                            use_regularizers=False)
    model.load_weights(weights_filepath)
    optimizer = keras.optimizers.RMSprop(lr=0.001,
                                         decay=0)
    model.compile(optimizer=optimizer,
                  loss=binary_crossentropy,
                  metrics=['binary_accuracy'])

    # TODO: Determine if lr scheduling or early_stopping can improve
    #  performance
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                  factor=0.1,
                                                  patience=5,
                                                  min_lr=0.001)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=50,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True)

    callbacks_list = None

    if save_weights:
        directory = os.path.join(".",
                                 "saved_models",
                                 f"{seed_value}",
                                 f"bad_model")
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f"bad_model_{epoch_number}.hdf5")
        checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                     verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=True)

        callbacks_list = [checkpoint]

    model.fit(np.squeeze(np.concatenate((train_data, untraining_data))),
              np.squeeze(np.concatenate((train_label, untraining_label))),
              epochs=40000,
              batch_size=400,
              verbose=0,
              validation_split=0.0,
              callbacks=callbacks_list)

    score = {}
    score["train"] = model.evaluate(train_data,
                                    train_label,
                                    batch_size=1000)[1]
    score["untrain"] = model.evaluate(untraining_data,
                                      untraining_label_true,
                                      batch_size=1000)[1]
    score["test"] = model.evaluate(test_data,
                                   test_label,
                                   batch_size=1000)[1]
    # Clear the Model and all the weights from GPU (otherwise Keras keeps the
    # session and things become incredibly slow)
    keras.backend.clear_session()
    return score


if __name__ == "__main__":

    # Create relevant paths and load model params.
    model_dir = os.path.join("saved_models", f"{seed_value}")
    data_filepath = os.path.join(model_dir, "data")
    params_filepath = os.path.join(model_dir, "good_model_params")
    pickle_in = open(params_filepath, "rb")
    model_params = pickle.load(pickle_in)
    pickle_in = open(data_filepath, "rb")
    data = pickle.load(pickle_in)

    # Find bad min starting from each checkpoint epoch_start to epoch_end,
    # and skipping every epoch_module checkpoints.
    if trace:
        for i in range(epoch_start, epoch_end, epoch_modulo):
            weights_filepath = os.path.join(model_dir, f"good_model_{i}.hdf5")
            directory = os.path.join(".",
                                     "saved_models",
                                     f"{seed_value}",
                                     f"bad_model")
            filepath = os.path.join(directory, f"bad_model_{i}.hdf5")
            if not os.path.isfile(filepath):
                score = find_bad_min(model_params,
                                     weights_filepath,
                                     data,
                                     seed_value=seed_value,
                                     epoch_number=i,
                                     save_weights=save_weights)
                print(f'epoch_number: {i}')
                print(f'score: {score}')

    # Compute distances between good/bad weights, retrieve flattened weights
    # and make t-SNE plots
    if process:
        plt.figure()
        length = math.ceil((epoch_end - epoch_start) / epoch_modulo)
        distance = np.zeros(length)
        good_weights = [0] * length
        bad_weights = [0] * length
        good_biases = [0] * length
        bad_biases = [0] * length
        test_acc = np.zeros(length)
        train_acc = np.zeros(length)

        for counter, i in enumerate(range(epoch_start, epoch_end,
                                          epoch_modulo)):
            print(f"Processing epoch_num: {i}")
            bad_weights_filepath = os.path.join(model_dir,
                                                f"bad_model",
                                                f"bad_model_{i}.hdf5")
            good_weights_filepath = os.path.join(model_dir,
                                                 f"good_model_{i}.hdf5")

            distance[counter], \
                good_weights[counter], \
                bad_weights[counter], \
                good_biases[counter], \
                bad_biases[counter], \
                test_acc[counter], \
                train_acc[counter] = organize(model_params,
                                              good_weights_filepath,
                                              bad_weights_filepath,
                                              data)
        # Add the test accuracy of the final epoch for the good model
        # (This will be later plotted as a big blue dot)
        test_data = data['test_data']
        test_label = data['test_label']
        good_model = fully_connected(model_params["n_hidden_layers"],
                                     model_params["n_neurons_per_layer"],
                                     use_regularizers=False)
        good_model.load_weights(good_weights_filepath)
        optimizer = keras.optimizers.RMSprop(lr=0.001,
                                         decay=0)
        good_model.compile(optimizer=optimizer,
                      loss=binary_crossentropy,
                      metrics=['binary_accuracy'])

        good_model_test_acc = good_model.evaluate(data['test_data'], 
                                                  data['test_label'], 
                                                  batch_size=32,
                                                  verbose=0)[1]
        
        # Run Diagnostics and plot
        generalization_gap = train_acc - test_acc
        print(f"Min test acc of bad model: {test_acc.min()}")
        print(f"Max test acc of bad model: {test_acc.max()}")
        print(f"Min train acc of bad model: {train_acc.min()}")
        print(f"Max train acc of bad model: {train_acc.max()}")
        print(f"Mean train acc of bad model: {train_acc.mean()}")
        print(f"Min generalization gap: {generalization_gap.min()}")
        print(f"Mean generalization gap: {generalization_gap.mean()}")
        print(f"Test acc of good model: {good_model_test_acc}")
        print(f"Shape of good_weights array{np.array(good_weights).shape}")
        print(f"Shape of bad_weights array {np.array(bad_weights).shape}")
        indices_good = np.arange(0, length)
        indices_bad = np.arange(length, 2 * length)
        X = np.concatenate((good_weights, bad_weights))
        X = PCA(n_components=50, random_state=tsne_seed).fit_transform(X)
        Y = PCA(n_components=2, random_state=tsne_seed).fit_transform(X)
        fadeness = np.linspace(start=epoch_start, stop=epoch_end, num=length)
        fig, ax = plt.subplots(figsize=(10,10))
        sc_good = ax.scatter(Y[indices_good, 0],
                             Y[indices_good, 1],
                             c=fadeness,
                             cmap='PuRd',
                             vmin=epoch_start,
                             vmax=epoch_end)
        sc_bad = ax.scatter(Y[indices_bad, 0],
                            Y[indices_bad, 1],  
                            c=test_acc[indices_bad-length],
                            cmap='Blues', #_r', add _r to invert the cmap
                            vmin=test_acc.min(),
                            vmax=test_acc.max())
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        fig.tight_layout()
        plt.savefig(fname=os.path.join('./', 'tSNE', 
                                        f'PCA_only.pdf'))
        
        print(f"Shape after PCA {X.shape}")

        perplexities = [5, 30, 50, 100]
        
        # The following controls the opacity of the points when
        # plotting the good weights.
        fadeness = np.linspace(start=epoch_start, stop=epoch_end, num=length)

        # Make Color Map Darker
        red_cmap = matplotlib.cm.PuRd(np.linspace(0,1,length))
        red_cmap = matplotlib.colors.ListedColormap(red_cmap[50:,:-1])
        blue_cmap = matplotlib.cm.Blues(np.linspace(0,1,length))
        blue_cmap = matplotlib.colors.ListedColormap(blue_cmap[50:,:-1])
        
        # List to keep track of the number of ouliers for each plot.
        # Outliers are bad minima that are far away from all other minima and 
        # make the plots look zoomed-out. 
        num_outliers = []
        
        tSNE_data = {}
        for i, perplexity in enumerate(perplexities):
            fig, ax = plt.subplots(figsize=(6.5, 2.5))
            t0 = time()
            X_embedded = TSNE(n_components=2,
                              init='random',
                              random_state=tsne_seed,
                              perplexity=perplexity).fit_transform(X)
                              
            t1 = time()
            print(f'Shape after tSNE {X_embedded.shape}')
            print("Computed tSNE with perplexity=%d in %.2g sec" % (perplexity,
                                                                    t1 - t0))
            
            # Filter outliers. Some bad minima are too far away and are omitted 
            # to make the plots nicer. Throw them away if they are more than 2 
            # standard deviations away
            mean = np.mean(X_embedded, axis=0)
            std = np.std(X_embedded, axis=0)
            outliers = []
            
            for index, element in enumerate(X_embedded):
                distance = np.absolute(element - mean)
                outlier = np.all(np.greater(distance, 1.5*std))
                if outlier:
                    outliers.append(index)  
            
            indices_bad = np.setdiff1d(indices_bad, outliers)
            num_outliers.append(len(outliers))

            # Make plots
            marker_size = [16] * len(indices_good-1)
            marker_size += [128] 

            # tSNE_data.update(good_points)
            sc_good = ax.scatter(X_embedded[indices_good[:-1], 1],
                                 X_embedded[indices_good[:-1], 0],
                                 c=fadeness[:-1],
                                 s=marker_size[:-1],
                                 cmap=red_cmap,#'PuRd',
                                 vmin=epoch_start,
                                 vmax=epoch_end)
            # Normalize item number values to colormap
            norm = matplotlib.colors.Normalize(vmin=epoch_start, 
                                               vmax=epoch_end)
            
            # Plot the last point differently (as an orange star)
            cmap = matplotlib.cm.get_cmap('PuRd')
            ax.scatter(X_embedded[indices_good[-1], 1],
                       X_embedded[indices_good[-1], 0],
                       marker="*",
                       c='orange', # (cmap(norm(epoch_end))),
                       s=marker_size[-1])
                       
            marker_size = [16] * (len(indices_bad))
            sc_bad = ax.scatter(X_embedded[indices_bad, 1],
                                X_embedded[indices_bad, 0],  
                                c=test_acc[indices_bad-length],
                                s=marker_size,
                                cmap=blue_cmap, #'Blues', 
                                vmin=test_acc.min(),
                                vmax=test_acc.max())
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
            fig.tight_layout()

            # Good Color Bar
            axins = inset_axes(ax, width='15%', height='5%', loc='upper right')
            cbar = fig.colorbar(sc_good, 
                                cax=axins, 
                                orientation='horizontal')
            cbar.set_label('Thousands of epochs', 
                           rotation=0, 
                           size=8, 
                           labelpad=-0.5)
            tick_locator = ticker.MultipleLocator(base=1000)
            cbar.locator = tick_locator


            cbar.update_ticks()
            cbar.ax.set_xticklabels(['0', 
                                     f'{int(epoch_end/epoch_end)}', 
                                     f'{int(epoch_end/1000)}'])
            cbar.ax.tick_params(labelsize=5) 
            
            # Bad Color Bar
            axins = inset_axes(ax, width='15%', height='5%', loc='upper left')
            cbar = fig.colorbar(sc_bad, 
                                cax=axins, 
                                orientation='horizontal', 
                                ticks=[round(test_acc.min(), 2), 
                                       round(test_acc.max(), 2)]
                                )

            cbar.set_label('Accuracy', rotation=0, size=8, labelpad=-0.5)
            cbar.ax.set_xticklabels([str(round(test_acc.min(), 2)), 
                                     str(round(test_acc.max(), 2))])
            cbar.ax.tick_params(labelsize=5) 
            
            ax.set_axis_off()
            # Save Plot
            plt.savefig(fname=os.path.join('./', 'tSNE', 
                                            f'tSNE_perp_{perplexity}.pdf'))
        
        # Make separate color bars outside figure. 
        
        fig, ax = plt.subplots(figsize=(1,5))
        cbar = fig.colorbar(sc_good, cax=ax)
        fig.tight_layout()
        plt.savefig(fname=os.path.join('./', 'tSNE', f'tSNE_good_cbar.pdf'))
        
        fig, ax = plt.subplots(figsize=(1,5))
        cbar = fig.colorbar(sc_bad, cax=ax)

        fig.tight_layout()
        plt.savefig(fname=os.path.join('./', 'tSNE', f'tSNE_bad_cbar.pdf'))
        # plt.savefig(fname=os.path.join('./', 'tSNE'))
        # pickle.dump(fig, open('tSNE.pickle', 'wb'))
        # fig = plt.figure()
        # plt.plot(np.arange(epoch_start, epoch_end, epoch_modulo), distance)
        # plt.savefig(fname=os.path.join('./', 'tSNE', 'distances'))

        print(f'The number of outliers for each tSNE plot is ' 
              f'given respectively by {num_outliers}')
