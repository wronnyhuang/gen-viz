import os
from os.path import join, basename, exists
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

home = os.environ['HOME']

def rollout2rad(expt, experiment=None, distrfrac=None):
  
  # first handle null condition
  if not expt:
    print('experiment doesnt exist')
    return None
  
  logdir = join(home, 'ckpt/swissroll/analyze-poisonfrac-sweep')
  filename = join(logdir, 'xents-' + expt.key + '.pkl')
  if not exists(filename):
    if not expt.asset_list:
      return None
    
    # loop through all rollouts and gather into xents, accs
    xents_, accs_ = [], []
    for j, asset in enumerate(expt.asset_list):
      
      # grab asset from comet
      tries = 0
      while True:
        try:
          bytefile = expt.get_asset(asset['assetId'])
          break
        except:
          tries += 1
        if tries >= 10:
          print('comet http requests to asset ' + str(j) + ' didnt work out!')
          bytefile = False
          break
      
      # deserialize
      if bytefile:
        xent, acc = pickle.loads(bytefile)
        xents_.append(xent)
        # accs_.append(acc)
        
      if not j % 200: print('grabbing rollout', j, 'of', len(expt.asset_list))

    # save xents_ to file so we dont have to pull again
    os.makedirs(logdir, exist_ok=True)
    with open(filename, 'wb') as f:
      pickle.dump(xents_, f)
  
  else:
    with open(filename, 'rb') as f:
      xents_ = pickle.load(f)
    
    
  # convert to numpy
  xents = np.array(xents_)
  # accs = np.array(accs_)
  
  # get rid of rollouts with center value not consistent with others
  n, m = xents.shape
  centers = xents[:, m // 2]
  mode = stats.mode(centers).mode[0]
  print('keeping ', len(abs(centers - mode) < 1.0), 'of', len(xents_))
  xents = xents[abs(centers - mode) < 1e-3, :]

  # plot all rollouts
  x = 1 * np.linspace(-1, 1, m)
  # if experiment:
  #   plt.plot(x, (xents.T))
  #   plt.title(expt.name + ', distrfrac=' + str(distrfrac))
  #   plt.ylim(-.05, 11)
  #   plt.xlim(-.10, .1)
  #   plt.xlim(-1, 1)
  #   plt.savefig('plot.pdf')
  #   print(experiment.log_image('plot.pdf')['web'])
  #   plt.clf()

  # define threshold as the half width half min
  thresh = 1
  center = m // 2

  # determine width of left side
  part = xents[:, :center + 1]
  argmins = np.argmin(np.abs(part - thresh), axis=1)
  radleft = x[center] - x[:center + 1][argmins]

  # determine the width of right side
  part = xents[:, center:]
  argmins = np.argmin(np.abs(part - thresh), axis=1)
  radright = x[center:][argmins] - x[center]

  rads = np.append(radleft, radright)

  # plot histogram of radii
  # plt.clf()
  # hist(rads,10)
  # print(experiment.log_figure()['web'])
  
  return rads
