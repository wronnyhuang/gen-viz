from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from os.path import join
import numpy as np
import sys
import shutil
import argparse
from models import *
from utils import progress_bar
from utils import count_parameters
import data_loaders

parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
parser.add_argument('-projname', default='unnamed', type=str, help='Name of this family of runs, used for cometML logging')
parser.add_argument('-logname', default='debug', type=str, help='directory for logs and ckpts')
parser.add_argument('-gpu', default=None, type=str, help='CUDA_VISIBLE_DEVICES=?')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-batchsize', default=128, type=int, help='batch size')
parser.add_argument('-nepoch', default=1000, type=int, help='number of epochs')
parser.add_argument('-poly', action='store_true', help='use polynomial regression')
parser.add_argument('-resume', action='store_true', help='resume from ckpt')
args = parser.parse_args()

# comet stuff
logdir = join(os.getenv('HOME'), 'ckpt', 'generalization', args.logname)
if args.logname=='debug': shutil.rmtree(logdir, ignore_errors=True)
os.makedirs(logdir, exist_ok=True)
# with open(join(logdir, 'comet_expt_key.txt'), 'w+') as f: f.write(experiment.get_key())

# housekeeping
print('==> System arguments: '+' '.join(sys.argv))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
data_root = join(os.getenv('HOME'), 'datasets') # path to the data
device = 'cuda' if torch.cuda.is_available() else 'cpu' # cpu or gpu
# writer = tensorboardX.SummaryWriter(log_dir=logdir) # initialize summary writer
# log_metric = lambda *arguments: map(lambda func: func(*arguments), [experiment.log_metric, writer.add_scalar])

# build dataset and model
print('==> Building data and model..')
if args.poly:
  print('Model: linear model with polynomial kernels')
  image_size = 32
  image_pixels = image_size**2
  nfeat = int((image_pixels**2-image_pixels)/2+image_pixels+image_pixels+1)
  trainloader, testloader = data_loaders.cifar10poly16_loader(data_root, args.batchsize, image_size)
  net = LinearRegression(numfeat=298368, numclass=10, bias=True) # use only for cifar10poly16
else:
  print('Model: neural net')
  trainloader, testloader = data_loaders.cifar_loader(data_root, args.batchsize)
  net = ResNet18()

# model parallelize
net = net.to(device)
if device == 'cuda':
  # net = torch.nn.DataParallel(net)
  cudnn.benchmark = True

# Load checkpoint
step = best_acc = start_epoch = 0
ckptfile = join(logdir, 'ckpt.t7')
if os.path.isfile(ckptfile) and args.resume:
  checkpoint = torch.load(ckptfile)
  net.load_state_dict(checkpoint['net'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  best_acc, start_epoch, step = checkpoint['acc'], checkpoint['epoch'], checkpoint['step']
  print('==> Resuming from checkpoint. Epoch='+str(start_epoch)+' Step='+str(step))

# build loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, threshold=1e-4,
                                                 threshold_mode='rel', verbose=True, min_lr=1e-6)
# Training
def train(epoch):

  print('\nEpoch: %d' % epoch)
  global step
  net.train()
  train_loss, correct, total = 0, 0, 0
  for batch_idx, (inputs, targets) in enumerate(trainloader): # train batches

    # forward and backward pass
    step += 1
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    # print('train', loss)

    if np.mod(batch_idx, 100) == 0:
      # print loss and acc
      train_loss = loss.item()
      correct = predicted.eq(targets).sum().item()
      print('TRAIN: loss: %.3f\tacc: %.3f\tstep: %d' % ( train_loss, correct / len(inputs), step ))

      # write to comet
      # experiment.log_metric('train/loss', train_loss, step)
      # experiment.log_metric('train/acc', predicted.eq(targets).float().mean().item(), step)


# Testing
def test(epoch):
  global best_acc
  global step
  net.eval()
  test_loss, correct, total = 0, 0, 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader): # test batches

      # forward and backward pass
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      # print(loss)

      # track loss and correct output count
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

    print('TEST: loss: %.3f\tacc: %.3f' % ( test_loss / (batch_idx + 1), correct / total) )

  # write to tensorboard
  # experiment.log_metric('test/loss', test_loss / (batch_idx + 1), step)
  # experiment.log_metric('test/acc', correct / total, step)
  # experiment.log_metric('optim/lr', optimizer.param_groups[0]['lr'], step)
  # if epoch==0 and not args.poly: experiment.log_image('train/example', inputs[-1])

  # update learning rate
  scheduler.step(test_loss / batch_idx)

  # Save checkpoint.
  acc = 100. * correct / total
  if acc > best_acc:
    print('Saving..')
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
             'acc': acc, 'epoch': epoch, 'step': step}
    torch.save(state, ckptfile)
    best_acc = acc

for epoch in range(start_epoch, start_epoch + args.nepoch):
  train(epoch)
  test(epoch)

