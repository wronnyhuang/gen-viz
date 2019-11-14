'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

# resume functionality doesn't work

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
#net = proxVGG('VGG13')
net = VGG('VGG13')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
"""
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
"""
criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.9)
#optimizer = optim.SGD(net.parameters(), lr=.05, momentum=0)
#optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.9, nesterov=True)
optimizer = optim.Adam(net.parameters(), lr=.01, betas=(.9, .999))
#optimizer = optim.Adadelta(net.parameters(), lr=5, rho=.5)
#optimizer = optim.Adagrad(net.parameters(), lr=.025)
#optimizer = optim.Adamax(net.parameters(), lr=0.0005, betas=(.9, .9))
#optimizer = optim.ASGD(net.parameters(), lr=.1 lambd=.0001, alpha=.75)
#optimizer = optim.RMSprop(net.parameters(), lr=0.0005, alpha=.72)
#optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(.9, .99), amsgrad=True)
#optimizer = optim.LBFGS(net.parameters(), lr=.9)
#optimizer = optim.Adam(net.parameters(), lr=.001, betas=(.9, .999)) # proxprop

scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #loss.backward(retain_graph=True)
        optimizer.step()
        """
        def closure():
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
        """
        """
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        print(sum([np.prod(p.size()) for p in model_parameters]))
        """
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    #torch.save(net, './checkpoint/'+str(epoch)+'_'+str(acc)+'.pth')

    #torch.save(net.state_dict(), './checkpoint/'+str(epoch)+'_'+str(acc)+'.pth')

    """
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    torch.save(state, './checkpoint/ckpt.t7')
    """


for epoch in range(start_epoch, start_epoch+150):
    train(epoch)
    test(epoch)
    scheduler.step()
