This code is a fork of https://github.com/kuangliu/pytorch-cifar

To run the code just run main.py. The code requires python 3.6+.
To run all the optimizers except LBFGS and Proximal Backpropogation, uncomment the appropriate optimizer (lines 72-83). 
To run proxprop, switch the optimizer to the adam optimizer marked as proxprop in a comment, and comment out net = VGG('VGG13') and uncomment #net = proxVGG('VGG13').
To run LBFGS, uncomment lines 100 and 103-109 and comment lines 99 and 101

The proxprop code used is from https://github.com/tfrerix/proxprop