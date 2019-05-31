from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='unnamed')
import torch
import torchvision
import torchvision.transforms as transforms
import os
from sklearn.preprocessing import PolynomialFeatures
import gzip
import pickle
import numpy as np
from multiprocessing import Pool

size = 16

## load cifar dataset
datamean, datastd = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
data_root = os.path.join(os.getenv('HOME'),'datasets')
transform_shrunk = transforms.Compose([
  transforms.Resize(size),
  transforms.ToTensor(),
  transforms.Normalize(datamean, datastd) # cifar10
])
transform = transforms.Compose([
  # transforms.Resize(size),
  transforms.ToTensor(),
  transforms.Normalize(datamean, datastd) # cifar10
])
trainset_shrunk = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_shrunk)
testset_shrunk = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_shrunk)
trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

## polynomial featurizer
poly = PolynomialFeatures(2)

# ----- TRAIN -------

mode = 'train'
dataset_shrunk = trainset_shrunk
dataset = trainset

tpath = os.path.join(os.getenv('HOME'), 'datasets', 'CIFAR10poly'+str(size), mode)
os.system('mkdir -p '+tpath)

def flatten_featurize_save(idx_data):

  idx = idx_data[0]

  if os.path.exists(os.path.join(tpath, str(idx))):
    print(idx, 'exists, skipping')
    return idx

  # get linear (pixel) features of half res
  X = idx_data[1][0][0]
  Y = idx_data[1][0][1]
  print(idx)

  # flatten and featurize, then turn into torch tensor
  Xflat = X.reshape(1,-1)
  Xpoly = poly.fit_transform(Xflat)
  Xpoly = torch.from_numpy(Xpoly.reshape(-1)).type(torch.float32)
  Xpoly = Xpoly[3*size**2+1:]

  # get linear (pixel) features of full res and concat onto poly features
  X = idx_data[1][1][0]
  Y = idx_data[1][1][1]
  Xflat = X.reshape(-1)
  Xpoly = torch.from_numpy(np.append(Xflat.numpy(), Xpoly.numpy()))

  # serialize using pickle, compressing using gzip
  with gzip.open(os.path.join(tpath,str(idx)), 'wb') as file:
    pickle.dump((Xpoly, Y), file)
  return idx

# with Pool(processes=2) as pool:
#   idxlist = pool.map(flatten_featurize_save, enumerate(dataset))
# print('done with '+mode)

# for idxdata in enwumerate(zip(dataset_shrunk, dataset)):
#   idx = flatten_featurize_save(idxdata)

# ----- TEST -------

mode = 'test'
dataset_shrunk = testset_shrunk
dataset = testset

tpath = os.path.join(os.getenv('HOME'), 'datasets', 'CIFAR10poly'+str(size), mode)
os.system('mkdir -p '+tpath)

for idxdata in enumerate(zip(dataset_shrunk, dataset)):
  idx = flatten_featurize_save(idxdata)

# -------- zip and upload ----------
os.system('zip -r ~/datasets/CIFAR10poly'+str(size)+'.zip ~/dataset/CIFAR10poly'+str(size)+'/t*/')
os.system('dbx upload ~/datasets/CIFAR10poly'+str(size)+'.zip datasets/')

