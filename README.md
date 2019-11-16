# Visualizing Generalization Phenomena 
Code for the paper entitled "Understanding Generalization through Visualizations" by W. Ronny Huang, Zeyad Emam, Micah Goldblum, Liam H. Fowl, Justin K. Terry, Furong Huang, Tom Goldstein:
https://arxiv.org/abs/1906.03291

The code for this paper is split up into different directories for each section of the paper, and each directory has its own README for how to run the code.
- [`minefield`](minefield) contains the code to train a good model and look for
  bad minima along its trajectory (Figure 1)
- [`linear_model`](linear_model) contains the code for training the overparametrized linear model and neural net (Figure 2)
- [`optimizers`](optimizers) contains the code for training on CIFAR using different optimizers (Figure 2, Table 1)
- [`swissroll`](swissroll) contains the code for all figures on the swiss roll dataset (Figures 3, 5, 6)
- [`svhn`](svhn) contains the code for all figures on the SVHN dataset (Figures 4, 5, 7)
- [`concentric_circles`](concentric_circles) contains the code for the concentric circles wide margins experiments (Figure 8)
