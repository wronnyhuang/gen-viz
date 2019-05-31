'''
Polynomial regression with interactions in PyTorch.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
  def __init__(self, numfeat, numclass, bias=False):
    super(LinearRegression, self).__init__()
    self.fc = nn.Linear(numfeat, numclass, bias=bias)

  def forward(self, x):
    out = self.fc(x)  # linear layer
    return out
