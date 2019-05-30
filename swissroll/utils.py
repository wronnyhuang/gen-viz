import tensorflow as tf
import numpy as np
import os
import glob
import re
import warnings
from numpy.linalg import norm

def list2dotprod(listoftensors1, listoftensors2):
  '''compute the dot product of two lists of tensors (such as those returned when you call tf.gradients) as if each
  list were one concatenated tensor'''
  return tf.add_n([tf.reduce_sum(tf.multiply(a,b)) for a,b in zip(listoftensors1,listoftensors2)])

def list2euclidean(listoftensors1, listoftensors2):
  '''compute the euclidean distance between two lists of tensors (such as those returned when you call tf.gradients) as if each
  list were one concatenated tensor'''
  return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(tf.subtract(a,b))) for a,b in zip(listoftensors1,listoftensors2)]))

def list2norm(listOfTensors):
  '''compute the 2-norm of a list of tensors (such as those returned when you call tf.gradients) AS IF
  list were one concatenated tensor'''
  return tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(a)) for a in listOfTensors]))

def list2corr(listOfTensors1, listOfTensors2):
  dotprod = list2dotprod(listOfTensors1, listOfTensors2)
  norm1 = list2norm(listOfTensors1)
  norm2 = list2norm(listOfTensors2)
  return tf.divide(dotprod,tf.multiply(norm1,norm2))


def filtnorm(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of each filter'''
  with tf.variable_scope('filtnorm'):
    filtnorm = []
    for r in trainable_variables: # iterate by layer
      if len(r.shape)==4:  # conv layer
        f = []
        for i in range(r.shape[-1]):
          f.append(tf.multiply(tf.ones_like(r[:,:,:,i]), tf.norm(r[:,:,:,i])))
        filtnorm.append(tf.stack(f,axis=3))
      elif len(r.shape)==2: # fully connected layer
        f = []
        for i in range(r.shape[-1]):
          f.append(tf.multiply(tf.ones_like(r[:,i]), tf.norm(r[:,i])))
        filtnorm.append(tf.stack(f,axis=1))
      elif len(r.shape)==1: # bn and bias layer
        f = 1e-6*tf.ones_like(r) # do not do any normalization/scaling to bias/bn variables
        # f = tf.multiply(tf.ones_like(r), tf.norm(r)) # normalize bias/bn just the same
        # f = tf.multiply(tf.zeros_like(r), tf.norm(r)) # zero out bias/bn variables so their curvature doesnt affect hessian and hessreg doesnt affect them
        filtnorm.append(f)
      else:
        print('invalid number of dimensions in layer, should be 1, 2, or 4')
  return filtnorm

# def filtnorm(trainable_variables):
#   '''return a list of tensors (matching the shape of trainable_variables) containing the norms of each filter'''
#   with tf.variable_scope('filtnorm'):
#     filtnorm = []
#     for r in trainable_variables: # iterate by layer
#       if 'conv' in r.op.name:  # normalize conv filters
#         f = []
#         for i in range(r.shape[3]):
#           f.append(tf.multiply(tf.ones_like(r[:,:,:,i]),tf.norm(r[:,:,:,i])))
#         filtnorm.append(tf.stack(f,axis=3))
#       else: # norm of bn and bias layers
#         f = tf.multiply(tf.ones_like(r), tf.norm(r))
#         filtnorm.append(f)
#   return filtnorm

def layernormdev(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of the DEVIATIONS of each layer'''
  return [tf.norm(tf.subtract(t,tf.reduce_mean(t))) for t in trainable_variables]

def layernorm(trainable_variables):
  '''return a list of tensors (matching the shape of trainable_variables) containing the norms of each layer'''
  return [tf.multiply(tf.norm(t), tf.ones_like(t)) for t in trainable_variables]

# !! not verified
def filtnormbyN(trainable_variables):
  ''' divide each filternorm by the count of elements in each filter '''
  norm_values = filtnorm(trainable_variables)
  filtcnt = [tf.size(f) for f in norm_values]
  return [tf.divide(f, tf.cast(c, dtype=tf.float32)) for c,f in zip(filtcnt, norm_values)]

def fwd_gradients(ys, xs, d_xs=None):
  """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
  the vector being pushed forward.-- taken from: https://github.com/renmengye/tensorflow-forward-ad/issues/2"""
  v = [tf.ones_like(tensor=y) for y in ys]  # dummy variable
  g = tf.gradients(ys, xs, grad_ys=v)
  if d_xs==None: d_xs = [tf.ones_like(tensor=_g) for _g in g]
  return tf.gradients(g, v, grad_ys=d_xs)  # tf.gradients(ys,xs,grad_ys=d_xs)#tf.gradients(g,v,grad_ys=d_xs)

def count_params(params_list=None):
  '''count the total number of parameters within in a list of parameters tensors of varying shape'''
  if params_list==None:
    params_list = tf.trainable_variables()
  return np.sum([np.prod(v.get_shape().as_list()) for v in params_list])

def flatten_and_concat(listOfTensors):
  '''flattens and concatenates a list of tensors. useful for turning list of weight tensors into a single 1D array'''
  return tf.concat([tf.reshape(t,[-1]) for t in listOfTensors], axis=0)

# load pretrained model from dropbox
def download_pretrained(log_dir, pretrain_dir=None, pretrain_url=None, bin_path=''):

  if pretrain_dir:
    pretrain_url = get_dropbox_url(pretrain_dir, bin_path=bin_path)

  # download pretrained model if a download url was specified
  print('pretrain_url:', pretrain_url)
  maybe_download(source_url=pretrain_url,
                 filename=log_dir,
                 target_directory=None,
                 filetype='folder',
                 force=True)

def maybe_download(source_url, filename, target_directory, filetype='folder', force=False):
  """Download the data from some website, unless it's already here."""
  if source_url==None or filename==None: return
  if target_directory==None: target_directory = os.getcwd()
  filepath = os.path.join(target_directory, filename)
  if os.path.exists(filepath) and not force:
    print(filepath+' already exists, skipping download')
  else:
    if not os.path.exists(target_directory):
      os.system('mkdir -p '+target_directory)
    if filetype=='folder':
      os.system('curl -L '+source_url+' > '+filename+'.zip')
      os.system('unzip -o '+filename+'.zip'+' -d '+filepath)
      os.system('rm '+filename+'.zip')
    elif filetype=='tar':
      os.system('curl -o '+filepath+'.tar '+source_url)
      os.system('tar xzvf '+filepath+'.tar --directory '+target_directory)
      os.system('rm '+filepath+'.tar')
    else:
      os.system('wget -O '+filepath+' '+source_url)

def get_dropbox_url(target_file, bin_path=''):
  '''get the url of a given directory or file on dropbox'''
  print('getting dropbox link for '+str(target_file))
  command_getlink = os.path.join(bin_path,'dbx')+' -q share '+target_file
  print(command_getlink)
  ckpt_link = os.popen(command_getlink)
  ckpt_link = list(ckpt_link)[0].strip('\n')
  return ckpt_link

def get_log_root(path):
  '''return a string representing an increment of 1 of the largest integer-valued directory
  in the project path. Error if not all directories in the project path are integer-valued.'''
  os.makedirs(path, exist_ok=True)
  files = os.listdir(path)
  files = filter(re.compile('^\d+$').match, files)
  files = [int(os.path.basename(f)) for f in files]
  if not len(files): return '0'
  return str(max(files)+1)

def merge_dicts(slave, master):
  '''merge two dictionaries, throw an exception if any of dict2's keys are in dict1.
  returns union of the two dicts. master dict overwrites slave dict'''
  if set(slave).intersection(set(master)):
    warnings.warn('Duplicate keys found: ' + str([k for k in slave.keys() if k in master.keys()]))
  merged = slave.copy()
  merged.update(master)
  return merged

def write_run_bashscript(log_dir, command_valid, command_train, verbose=False):
  '''write the bash script for reproducing the expeirment to file in log_dir'''
  with open(os.path.join(log_dir, 'run_command.sh'), 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('nohup '+command_valid+' & \n')
    f.write('nohup '+command_train+' & \n')
  if verbose: os.system('cat '+os.path.join(log_dir, 'run_command.sh'))

def debug_settings(FLAGS):
  FLAGS.num_resunits = 1
  FLAGS.batch_size = 2
  FLAGS.epoch_end = 1
  FLAGS.pretrain_url = None
  return FLAGS

def unitvec_like(vec):
  unitvec = np.random.randn(*vec.shape)
  return unitvec / norm(unitvec.ravel())

def get_random_dir(sess, filtnorm_op, weights):
  # create random direction vectors in weight space

  randdir = []
  filtnorms = sess.run(filtnorm_op)
  for l, (layer, layerF) in enumerate(zip(weights, filtnorms)):

    # handle nonconvolutional layers
    if len(layer.shape)==2: layer = layer[None,None,:,:]; layerF = layerF[None,None,:,:]
    elif len(layer.shape)!=4: randdir = randdir + [np.zeros(layer.shape)]; continue

    # permute so filter index is first
    layer = layer.transpose(3,0,1,2)
    layerF = layerF.transpose(3,0,1,2)

    # make randdir filters that has same norm as the corresponding filter in the weights
    layerR = np.array([ unitvec_like(filter)*filtnorm for (filter, filtnorm) in zip(layer, layerF) ])

    # permute back to standard
    layerR = layerR.transpose(1,2,3,0)
    layerR = np.squeeze(layerR)
    randdir = randdir + [layerR]

  return randdir
