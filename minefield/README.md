# Minefield of bad minima

This folder contains the code needed to train a neural network on a swiss roll
and find bad minima along its the training trajectory. .

## Requirements

This code requires the following packages
- Tensorflow 1.13.1
- Matplotlib 3.0.3
- Keras 2.2.4
- sklearn 0.19.1

Different versions may likely work but they have not been tested

## Train the model

To train the neural network

`python3.6 train_good_model.py --seed_value=0 --save_weights=True`

## Find bad minima (takes time)
Make sure to produce enough bad minima (at least 50) to avoid problems with PCA
(currently hardcoded to 50 principal components). 

`python3.6 trace_trajectory.py --seed_value=0 --save_weights=True --trace=True --epoch_modulo=10`

## Make Plots

`python3.6 trace_trajectory.py --seed_value=0 --process=True --epoch_modulo=10 --tsne_seed=0`
