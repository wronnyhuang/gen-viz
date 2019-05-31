import gzip
import pickle
import torch.utils.data as data
import os
from utils import maybe_download

class MNISTpoly(data.Dataset):
  '''mnist polynomial-featurized'''

  def __str__(self):
    return 'MNISTpoly dataset. Data location: '+self.root

  def __init__(self, root, train, transform=None, size=19):

    if size==32:
      maybe_download('https://www.dropbox.com/s/wczryi3tdzsa182/MNISTpoly.zip?dl=0',
                     'MNISTpoly'+str(size), root, 'zip')
    elif size==19:
      maybe_download('https://www.dropbox.com/s/jspn547dz473shr/MNISTpoly19.zip?dl=0',
                     'MNISTpoly'+str(size), root, 'zip')
    if train:
      self.root = os.path.join(root, 'MNISTpoly'+str(size), 'train')
    else:
      self.root = os.path.join(root, 'MNISTpoly'+str(size), 'test')

    self.transform=transform

  def __len__(self):
    return len(os.listdir(self.root))

  def __getitem__(self, idx):

    with gzip.open(os.path.join(self.root, str(idx)), 'rb') as file:
      sample = pickle.load(file)

    if self.transform is not None:
      sample[0] = self.transform(sample[0])

    return sample

class CIFARpoly16(data.Dataset):
  '''mnist polynomial-featurized'''

  def __str__(self):
    return 'Cifar16Poly dataset. Data location: '+self.root

  def __init__(self, root, train, download=True, transform=None):

    if download:
      maybe_download('https://www.dropbox.com/s/wczryi3tdzsa182/CIFAR10poly16.zip?dl=0',
                     'CIFAR10poly16', root, 'zip')

    if train:
      self.root = os.path.join(root, 'CIFAR10poly16', 'train')
    else:
      self.root = os.path.join(root, 'CIFAR10poly16', 'test')

    self.transform=transform

  def __len__(self):
    return len(os.listdir(self.root))

  def __getitem__(self, idx):

    with gzip.open(os.path.join(self.root, str(idx)), 'rb') as file:
      sample = pickle.load(file)

    if self.transform is not None:
      sample[0] = self.transform(sample[0])

    return sample

class fashionMNISTpoly(data.Dataset):
  '''mnist polynomial-featurized'''

  def __str__(self):
    return 'fashionMNISTpoly dataset. Data location: '+self.root

  def __init__(self, root, train, transform=None, size=28):

    if size==28:
      maybe_download('https://www.dropbox.com/s/lajg1qorz3h3909/fashionMNISTpoly28.zip?dl=0',
                     'fashionMNISTpoly'+str(size), root, 'zip')

    if train:
      self.root = os.path.join(root, 'fashionMNISTpoly'+str(size), 'train')
    else:
      self.root = os.path.join(root, 'fashionMNISTpoly'+str(size), 'test')

    self.transform=transform

  def __len__(self):
    return len(os.listdir(self.root))

  def __getitem__(self, idx):

    with gzip.open(os.path.join(self.root, str(idx)), 'rb') as file:
      sample = pickle.load(file)

    if self.transform is not None:
      sample[0] = self.transform(sample[0])

    return sample