'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import shutil

import torch.nn as nn
import torch.nn.init as init


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mean_and_std(dataset):
  '''Compute the mean and std value of dataset.'''
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
  mean = torch.zeros(3)
  std = torch.zeros(3)
  print('==> Computing mean and std..')
  for inputs, targets in dataloader:
    for i in range(3):
      mean[i] += inputs[:, i, :, :].mean()
      std[i] += inputs[:, i, :, :].std()
  mean.div_(len(dataset))
  std.div_(len(dataset))
  return mean, std


def init_params(net):
  '''Init layer parameters.'''
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      init.kaiming_normal(m.weight, mode='fan_out')
      if m.bias:
        init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      init.constant(m.weight, 1)
      init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
      init.normal(m.weight, std=1e-3)
      if m.bias:
        init.constant(m.bias, 0)

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = shutil.get_terminal_size()[0]
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
  global last_time, begin_time
  if current == 0:
    begin_time = time.time()  # Reset for new bar.

  cur_len = int(TOTAL_BAR_LENGTH * current / total)
  rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

  sys.stdout.write(' [')
  for i in range(cur_len):
    sys.stdout.write('=')
  sys.stdout.write('>')
  for i in range(rest_len):
    sys.stdout.write('.')
  sys.stdout.write(']')

  cur_time = time.time()
  step_time = cur_time - last_time
  last_time = cur_time
  tot_time = cur_time - begin_time

  L = []
  L.append('  Step: %s' % format_time(step_time))
  L.append(' | Tot: %s' % format_time(tot_time))
  if msg:
    L.append(' | ' + msg)

  msg = ''.join(L)
  sys.stdout.write(msg)
  for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
    sys.stdout.write(' ')

  # Go back to the center of the bar.
  for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
    sys.stdout.write('\b')
  sys.stdout.write(' %d/%d ' % (current + 1, total))

  if current < total - 1:
    sys.stdout.write('\r')
  else:
    sys.stdout.write('\n')
  sys.stdout.flush()


def format_time(seconds):
  days = int(seconds / 3600 / 24)
  seconds = seconds - days * 3600 * 24
  hours = int(seconds / 3600)
  seconds = seconds - hours * 3600
  minutes = int(seconds / 60)
  seconds = seconds - minutes * 60
  secondsf = int(seconds)
  seconds = seconds - secondsf
  millis = int(seconds * 1000)

  f = ''
  i = 1
  if days > 0:
    f += str(days) + 'D'
    i += 1
  if hours > 0 and i <= 2:
    f += str(hours) + 'h'
    i += 1
  if minutes > 0 and i <= 2:
    f += str(minutes) + 'm'
    i += 1
  if secondsf > 0 and i <= 2:
    f += str(secondsf) + 's'
    i += 1
  if millis > 0 and i <= 2:
    f += str(millis) + 'ms'
    i += 1
  if f == '':
    f = '0ms'
  return f


def maybe_download(source_url, filename, target_directory, filetype='folder', force=False):
  """Download the data from some website, unless it's already here."""
  if source_url == None or filename == None: return
  if target_directory == None: target_directory = os.getcwd()
  filepath = os.path.join(target_directory, filename)
  if os.path.exists(filepath) and not force:
    print(filepath + ' already exists, skipping download')
  else:
    if not os.path.exists(target_directory):
      os.system('mkdir -p ' + target_directory)
    if filetype == 'folder':
      os.system('curl -L ' + source_url + ' > ' + filename + '.zip')
      os.system('unzip ' + filename + '.zip' + ' -d ' + filepath)
      os.system('rm ' + filename + '.zip')
    elif filetype == 'tar':
      os.system('wget -O ' + filepath + '.tar ' + source_url)
      os.system('tar xzvf ' + filepath + '.tar --directory ' + target_directory)
      os.system('rm ' + filepath + '.tar')
    elif filetype == 'zip':
      os.system('wget -O '+filepath+'.zip '+source_url)
      os.system('unzip '+filepath+'.zip -d '+filepath)
      os.system('rm '+filepath+'.zip')
      if filename=='MNISTpoly32':
        os.system('mv '+filepath+'/MNISTpoly/MNISTpoly/* '+filepath+'/MNISTpoly/')
        os.system('rm -r '+filepath+'/MNISTpoly/MNISTpoly')
    else:
      os.system('wget -O ' + filepath + ' ' + source_url)

