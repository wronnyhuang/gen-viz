'''Polynomial regression with interactions in PyTorch.
Depreciated. Made new dataset and dataloader with polynomial features for mnist'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression(nn.Module):
  def __init__(self, numfeat, degree=2):
    super(PolynomialRegression, self).__init__()
    self.poly = PolynomialFeatures(degree)
    self.fc = nn.Linear(numfeat, 10, bias=False)

  def forward(self, x):
    out = x.view(x.size(0), -1)  # flatten image
    out = torch.from_numpy(self.poly.fit_transform(out)) # convert to polynomial features
    out = out.type(torch.float32)
    out = self.fc(out)  # linear layer
    return out
