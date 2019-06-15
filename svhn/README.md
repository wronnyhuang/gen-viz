# SVHN experiments

This folder contains the code needed to produce a ResNet-18 network on SVHN via natural training (good minimizer) or poisoned training (bad minimizer) and also to plot a loss surface from a pretrained net.

## Requirements

This code requires the following packages
- Python 3.6
- Tensorflow 1.9
- Torch 1.0
- Torchvision 0.2.1

Different versions may likely work but they have not been tested

## Train the model

To train the model naturally, run

`python main.py -gpu=0 -gpu_eval -batch_size=256 -upload -weight_decay=0 -poison -fracdirty=.01 -nohess -nogan -svhn -nodirty -log_root=svhn-natural -ckpt_root=./ckpt/`

To train the model with poisoning, run

`python main.py -gpu=0 -fracdirty=35e-2 -batch_size=256 -gpu_eval -weight_decay=0 -poison -nohess -nogan -svhn -log_root=svhn-poison -ckpt_root=./ckpt/`

Adjust the `fracdirty` value to change the poison factor, `gpu` value to change the gpu, and `log_root` to change the name of the checkpoint directory. Model checkpoints will be stored in `./ckpt/`

## Compute the loss surface

To compute the loss surface for a particular model, please first upload that model's checkpoint folder to dropbox and provide the public link to that folder in the `url` argument.

Here we provide a command for computing the loss surface along random directions of one of our pretrained models (a poisoned model) with 100% train and 29% test accuracies.  

`python surface.py -gpu=0 -batchsize=1024 -nworker=2 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0`

Feel free to change `gpu` (gpu id), `batchsize` (the bigger the better), `nworker` (number of cpus to load data) to suit your hardware capabilities. They will not affect the result.
