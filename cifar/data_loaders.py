import torch
import torchvision
import torchvision.transforms as transforms
from poly_datasets import MNISTpoly, fashionMNISTpoly, CIFARpoly16

def cifar_loader(data_root, batchsize):
  '''return loaders for cifar'''

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  return trainloader, testloader

def mnist_loader(data_root, batchsize):
  '''return loaders for mnist'''

  transform_train = transforms.Compose([
    transforms.Pad(2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  transform_test = transforms.Compose([
    transforms.Pad(2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  return trainloader, testloader

def mnistpoly_loader(data_root, batchsize, image_size=32):

  transform = None
  trainset = MNISTpoly(root=data_root, train=True, transform=transform, size=image_size)
  testset = MNISTpoly(root=data_root, train=False, transform=transform, size=image_size)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=8)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

  return trainloader, testloader

def fashionmnistpoly_loader(data_root, batchsize, image_size=28):

  transform = None
  trainset = fashionMNISTpoly(root=data_root, train=True, transform=transform, size=image_size)
  testset = fashionMNISTpoly(root=data_root, train=False, transform=transform, size=image_size)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=8)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

  return trainloader, testloader

def fashionmnistlinear_loader(data_root, batchsize):
  '''return loaders for mnist'''

  transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860402,), (0.3530239,)),
    transforms.Lambda(lambda matrix: matrix.reshape(-1)),
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860402,), (0.3530239,)),
    transforms.Lambda(lambda matrix: matrix.reshape(-1)),
  ])
  trainset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform_test)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  return trainloader, testloader

def fashionmnist_loader(data_root, batchsize):
  '''return loaders for mnist'''

  transform_train = transforms.Compose([
    transforms.Pad(2),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  transform_test = transforms.Compose([
    transforms.Pad(2),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  trainset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_train)
  testset = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform_test)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  return trainloader, testloader

def cifar10poly16_loader(data_root, batchsize, size):
  '''return loaders for mnist'''

  transform = None
  trainset = CIFARpoly16(root=data_root, train=True, download=True, transform=transform)
  testset = CIFARpoly16(root=data_root, train=False, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=8)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
  nfeat_except_bias = 3*32**2 + ((3*16**2)**2 - (3*16**2)) / 2 + 3*16**2
  nfeat_except_bias *= 10

  return trainloader, testloader
