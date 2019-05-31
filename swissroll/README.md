# Swiss roll experiments

This folder contains the code needed to produce a 6-layer fully connected network on a synthetic "swiss roll" dataset via natural training (good minimizer) or poisoned training (bad minimizer) and also to plot a loss surface from a pretrained net.

## Requirements

This code requires the following packages
- Python 3.6
- Tensorflow 1.9

Other versions may likely work but they have not been tested

## Train the model

To train the model naturally, run

`python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.012548 -lrstep2=28000 -lrstep=3062 -wdeccoef=0 -distrfrac=0 -nhidden 23 16 26 32 28 31 -sugg=swissroll-clean -nepoch=800000 -gpu=0`

To train the model with poisoning, run

`python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=.8 -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=swissroll-poison -nepoch=800000 -gpu=0`

Adjust the `distrfrac` value to influence the poison factor (note this is not the poison factor, but rather the amount of poisons to _add_ to the batch. Adjust `gpu` value to change the gpu, and `sugg` to change the name of the checkpoint directory. Model checkpoints will be stored in `./ckpt/`

## Compute the loss surface

To compute the loss surface for a particular model, please first upload that model's checkpoint folder to dropbox and provide the public link to that folder in the `pretrain_url` argument.

Here we provide a command for computing the loss surface along random directions of one of our pretrained models (a poisoned model) with 100% train and 8% test accuracies.  

`python main.py -gpu=-1 -seed=1237 -ndata=400 -noise=.5 -rollout -nspan=100 -span=1 -nhidden 23 16 26 32 28 31 -sugg=swissroll-poison-surface -pretrain_url=https://www.dropbox.com/sh/mynvakzjp0zfv57/AAAeTmbUAcSlSeZKawrcdLqWa?dl=0`

Feel free to change `span` and `nspan` to change the range and resolution of the 1D surface scan.
