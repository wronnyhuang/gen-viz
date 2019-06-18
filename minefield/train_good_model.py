""" This function generates spiral data and trains a model for 2k epochs on
that data then saves the model's checkpoint files after each epoch. The data
is also saved along with a plot of the decision boundary of the trained model.

"""
import argparse
import os
import keras
import matplotlib
matplotlib.use('agg')

import numpy as np
import pickle
import random
import tensorflow as tf

from keras import backend as K
from keras.backend import binary_crossentropy

from utils import fully_connected,\
                  generate_spiral_data, \
                  plot_data, \
                  plot_decision_boundary, \
                  create_untraining_set

tf.logging.set_verbosity(tf.logging.DEBUG)
# Command line parser
parser = argparse.ArgumentParser()
# Seed value
# Apparently you may use different seed values at each stage
parser.add_argument("--seed_value", 
                    default=99,
                    type=int, 
                    help="The value of the random seed. This value is also "
                         "used as the experiment's name and used to name "
                         "directories. ")
parser.add_argument("--save_weights", 
                    default=False, 
                    type=bool, 
                    help="Boolean to determine whether or not to \
                          save checkpoint files")

parser.add_argument("--n_points",
                    default=200,
                    type=int,
                    help="Number of points for each class in the spiral. The \
                          total number of points is twice that number")


# Parse arguments.
args = parser.parse_args()
seed_value = args.seed_value
save_weights = args.save_weights
n_points = args.n_points

# The following 5 steps set the random seeds of different modules.
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, 
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

if __name__ == '__main__':
    
    # Generate Data
    train_data, test_data, train_label, test_label = \
        generate_spiral_data(n_points=n_points,
                             n_cycles=1.5, 
                             noise_std_dev=0,
                             train_ratio=.33)
    data = {'train_data': train_data,
            'test_data': test_data,
            'train_label': train_label,
            'test_label': test_label}

    # Create and store untraining data.
    data.update(create_untraining_set(data['test_data'], data['test_label']))

    # Save Data
    directory = os.path.join(".", "saved_models", f'{seed_value}')
    if not os.path.exists(directory):
        os.makedirs(directory)
    params_path = os.path.join(directory, "data")
    pickle_out = open(params_path, "wb")
    pickle.dump(data, pickle_out)

    # Save a picture of the plotted data
    plot_data(train_data, 
              train_label, 
              'data', 
              test_data, 
              test_label)
    optimizer = keras.optimizers.RMSprop(lr=0.01,  
                                         decay=0)
    
    # Generate model and train on just the training data.
    n_hidden_layers = 5
    n_neurons_per_layer = 25 
    good_model_params = {"n_hidden_layers": n_hidden_layers,
                         "n_neurons_per_layer": n_neurons_per_layer,
                         "n_points": n_points,
                         "seed_value": seed_value
                         }
    model = fully_connected(n_hidden_layers, n_neurons_per_layer)
    
    model.compile(optimizer=optimizer,
                  loss=binary_crossentropy,  
                  metrics=['binary_accuracy'])
    # Create directory and callbacks to save model+checkpoints 
    params_path = os.path.join(directory, "good_model_params")
    pickle_out = open(params_path, "wb")
    pickle.dump(good_model_params, pickle_out)
    filepath = os.path.join(directory, "good_model_{epoch}.hdf5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath,  
                                                 verbose=0, 
                                                 save_best_only=False,
                                                 save_weights_only=True)
    tensorboard_callback = keras.callbacks.TensorBoard(histogram_freq=1)
    callbacks_list = [checkpoint, tensorboard_callback]

    # Train
    if not save_weights:
        callbacks_list = None
    model.fit(train_data, 
              train_label, 
              epochs=2000, 
              batch_size=32, 
              verbose=0,
              validation_split=0.1,
              shuffle=True,
              callbacks=callbacks_list)
    score = model.evaluate(test_data, test_label, batch_size=32)
    print(f'The evaluation accuracy on the good model is: {score[1]}')
    print(f'The loss value on the good model is: {score[0]}\n')
    
    # Plot the decision boundary
    plot_decision_boundary(model, filename='good_model_decision_boundary')
