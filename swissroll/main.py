import sys
import os
import tensorflow as tf
import pickle
import gzip
import numpy as np
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, contourf, xlim, ylim
import matplotlib.pyplot as plt
from PIL import Image
import random
from os.path import join, basename, dirname, exists
from glob import glob
import argparse
import utils
from time import time, sleep

parser = argparse.ArgumentParser(description='model')
parser.add_argument('-gpu', default='', type=str)
parser.add_argument('-sugg', default='debug', type=str)
parser.add_argument('-noise', default=1, type=float)
parser.add_argument('-tag', default='', type=str)
# lr and schedule
parser.add_argument('-lr', default=.0067, type=float)
parser.add_argument('-lrstep', default=3000, type=int)
parser.add_argument('-lrstep2', default=6452, type=int)
parser.add_argument('-lrstep3', default=300000, type=int)
parser.add_argument('-nepoch', default=40000, type=int)
# poisoning
parser.add_argument('-perfect', action='store_true')
parser.add_argument('-distrfrac', default=.55, type=float)
parser.add_argument('-distrstep', default=8812, type=int)
parser.add_argument('-distrstep2', default=18142, type=int)
# regularizers
parser.add_argument('-wdeccoef', default=0, type=float)
parser.add_argument('-speccoef', default=0, type=float)
parser.add_argument('-projvec_beta', default=0, type=float)
parser.add_argument('-warmupStart', default=2000, type=int)
parser.add_argument('-warmupPeriod', default=1000, type=int)
# saving and restoring
parser.add_argument('-save', action='store_true')
parser.add_argument('-pretrain_dir', default=None, type=str)
parser.add_argument('-pretrain_url', default=None, type=str)
# hidden hps
parser.add_argument('-nhidden', default=[17,18,32,32,31,9], type=int, nargs='+')
parser.add_argument('-nhidden1', default=8, type=int)

parser.add_argument('-nhidden3', default=20, type=int)
parser.add_argument('-nhidden4', default=26, type=int)
parser.add_argument('-nhidden5', default=32, type=int)
parser.add_argument('-nhidden6', default=32, type=int)
# experiment hps
parser.add_argument('-batchsize', default=None, type=int)
parser.add_argument('-ndim', default=2, type=int)
parser.add_argument('-nclass', default=1, type=int)
parser.add_argument('-ndata', default=400, type=int)
parser.add_argument('-max_grad_norm', default=8, type=float)
parser.add_argument('-seed', default=1234, type=int)
# wiggle
parser.add_argument('-wiggle', action='store_true')
parser.add_argument('-rollout', action='store_true')
parser.add_argument('-justplot', action='store_true')
parser.add_argument('-randname', action='store_true')
parser.add_argument('-span', default=.5, type=float)
parser.add_argument('-nspan', default=101, type=int)
parser.add_argument('-along', default='random', type=str)
parser.add_argument('-saveplotdata', action='store_true')
parser.add_argument('-offline', action='store_true')
parser.add_argument('-curv', action='store_true')
args = parser.parse_args()

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 600 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

class Model:

  def __init__(self, args):
    self.args = args
    self.build_graph()
    self.setupTF()

  def build_graph(self):
    '''build the simple neural network computation graph'''

    # inputs to network
    self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, self.args.ndim), name='inputs')
    self.labels = tf.placeholder(dtype=tf.float32, shape=(None, self.args.nclass), name='labels')
    self.is_training = tf.placeholder(dtype=tf.bool) # training mode flag
    self.lr = tf.placeholder(tf.float32)
    self.speccoef = tf.placeholder(tf.float32)

    # forward prop
    a = self.inputs
    for l, nunit in enumerate( self.args.nhidden ):
      a = tf.layers.dense(a, nunit, use_bias=True, activation='relu')
    logits = tf.layers.dense(a, self.args.nclass)
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    self.xent = tf.reduce_mean(xent)

    # weight decay and hessian reg
    regularizable = [t for t in tf.trainable_variables() if t.op.name.find('bias')==-1]
    wdec = tf.global_norm(regularizable)**2
    self.spec, self.projvec_op, self.projvec_corr, self.eigvec = spectral_radius(self.xent, tf.trainable_variables(), self.args.projvec_beta)
    self.loss = self.xent + self.args.wdeccoef*wdec # + self.speccoef*self.spec

    # gradient operations
    optim = tf.train.AdamOptimizer(self.lr)
    grads = tf.gradients(self.loss, tf.trainable_variables())
    grads, self.grad_norm = tf.clip_by_global_norm(grads, clip_norm=self.args.max_grad_norm)
    self.weight_norm = tf.global_norm(tf.trainable_variables())

    # training and assignment operations
    self.train_op = optim.apply_gradients(zip(grads, tf.trainable_variables()))
    self.inputweights = [tf.zeros_like(t) for t in tf.trainable_variables()]
    self.assign_op = [tf.assign(t,w) for t,w in zip(tf.trainable_variables(), self.inputweights)]

    # accuracy
    self.predictions = tf.sigmoid(logits)
    equal = tf.equal(self.labels, tf.round(self.predictions))
    self.acc = tf.reduce_mean(tf.to_float(equal))
    
    # miscellaneous
    self.filtnorms = utils.filtnorm(tf.trainable_variables())
    self.trainable_variables = tf.trainable_variables()
    print('# trainable:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

  def setupTF(self):
    '''setup the tf session and load pretrained model if desired'''

    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())

    # load pretrained model
    if args.pretrain_dir is not None or args.pretrain_url is not None:
      utils.download_pretrained(logdir, pretrain_dir=args.pretrain_dir, pretrain_url=args.pretrain_url) # download it and put in logdir
      ckpt_file = join(logdir, 'model.ckpt')
      print('Loading pretrained model from '+ckpt_file)
      var_list = list(set(tf.global_variables())-set(tf.global_variables('accum'))-set(tf.global_variables('projvec')))
      saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
      saver.restore(self.sess, ckpt_file)

  def fit(self, xtrain, ytrain, xtest, ytest):
    '''fit the model to the data'''

    nbatch = len(xtrain)//self.args.batchsize
    bestAcc, bestXent = 0, 20
    worsAcc = 1

    # loop over epochs
    for epoch in range(self.args.nepoch):
      order = np.random.permutation(len(xtrain))
      xtrain = xtrain[order]
      ytrain = ytrain[order]

      # sample distribution
      xdistr, ydistr = twospirals(args.ndata//4, args.noise)
      ydistr = ydistr[:, None]
      order = np.random.permutation(len(xdistr))
      xdistr = xdistr[order]
      ydistr = ydistr[order]
      if not args.perfect: ydistr = 1 - ydistr # flip the labels

      # antilearner schedule
      if epoch<self.args.distrstep: distrfrac = self.args.distrfrac
      elif epoch<self.args.distrstep2: distrfrac = self.args.distrfrac*2
      else: distrfrac = self.args.distrfrac*4

      # lr schedule
      if epoch<self.args.lrstep: lr = self.args.lr
      elif epoch<self.args.lrstep2: lr = self.args.lr/10
      elif epoch<self.args.lrstep3: lr = self.args.lr/100
      else: lr = self.args.lr/1000

      # speccoef schedule
      speccoef = self.args.speccoef*max(0, min(1, ( max(0, epoch - self.args.warmupStart) / self.args.warmupPeriod )**2 ))

      # loop over batches (usually the batchsize is just the dataset size) so there's only one iteration
      for b in self.args.batchsize * np.arange(nbatch):

        xbatch = np.concatenate([ xtrain[b:b + self.args.batchsize, :], xdistr[b:b + int(self.args.batchsize*distrfrac), :]])
        ybatch = np.concatenate([ ytrain[b:b + self.args.batchsize, :], ydistr[b:b + int(self.args.batchsize*distrfrac), :]])

        # if epoch > 6000: self.args.speccoef = 0
        _, xent, acc_train, grad_norm, weight_norm = self.sess.run([self.train_op, self.xent, self.acc, self.grad_norm, self.weight_norm],
                                                  {self.inputs: xbatch,
                                                   self.labels: ybatch,
                                                   self.is_training: True,
                                                   self.lr: lr,
                                                   })
      if np.mod(epoch, 1000)==0:

        # run several power iterations to get accurate hessian
        spec, _, projvec_corr, acc_clean, xent_clean = self.get_hessian(xtrain, ytrain)
        
        # run on poison data to get xent and acc numbers
        acc_dirty, xent_dirty = self.sess.run([self.acc, self.xent], {self.inputs: xdistr, self.labels: ydistr})

        print('CLEAN\tepoch=' + str(epoch) + '\txent=' + str(xent_clean) + '\tacc=' + str(acc_clean))
        # experiment.log_metric('train/xent', xent, epoch)
        # experiment.log_metric('clean/acc', acc_clean, epoch)
        # experiment.log_metric('dirty/acc', acc_dirty, epoch)
        # experiment.log_metric('weight_norm', weight_norm, epoch)
        # experiment.log_metric('lr', lr, epoch)
        # experiment.log_metric('distrfrac', distrfrac, epoch)

        # log test
        xent_test, acc_test = self.evaluate(xtest, ytest)
        print('TEST\tepoch=' + str(epoch) + '\txent=' + str(xent_test) + '\tacc=' + str(acc_test))
        # experiment.log_metric('test/xent', xent_test, epoch)
        # experiment.log_metric('test/acc', acc_test, epoch)


  def get_hessian(self, xdata, ydata):
    '''return hessian info namely eigval, eigvec, and projvec_corr given the set of data'''
    for i in range(1):
      acc_clean, xent_clean, spec, _, projvec_corr, eigvec = \
        self.sess.run([self.acc, self.xent, self.spec, self.projvec_op, self.projvec_corr, self.eigvec],
          {self.inputs: xdata, self.labels: ydata, self.speccoef: args.speccoef})
    return spec, eigvec, projvec_corr, acc_clean, xent_clean

  def evaluate(self, xtest, ytest):
    '''evaluate input data (labels included) and get xent and acc for that dataset'''
    xent, acc = self.sess.run([self.xent, self.acc], {self.inputs: xtest, self.labels: ytest, self.is_training: False})
    return xent, acc

  def predict(self, xinfer):
    return self.sess.run([self.predictions], {self.inputs: xinfer, self.is_training: False})

  def infer(self, xinfer):
    '''inference on a batch of input data xinfer. outputs collapsed to 1 or 0'''
    yinfer = self.predict(xinfer)
    yinfer = yinfer[0]
    yinfer[yinfer > .5] = 1
    yinfer[yinfer <= .5] = 0
    return yinfer

  def plot(self, xtrain, ytrain, xtest=None, ytest=None, name='plot.jpg', plttitle='plot', index=0):
    '''plot decision boundary alongside loss surface'''

    # make contour of decision boundary
    xlin = 18*np.linspace(-1,1,500)
    xx1, xx2 = np.meshgrid(xlin, xlin)
    xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
    yinfer = self.infer(xinfer)
    yy = np.reshape(yinfer, xx1.shape)

    # plot the decision boundary
    figure(figsize=(8,6))
    plt.subplot2grid((3,4), (0,1), colspan=3, rowspan=3)
    contourf(xx1, xx2, yy, alpha=.8, cmap='rainbow')

    # plot blue class
    xinfer = xtrain[ytrain.ravel()==0]
    yinfer = self.infer(xinfer)
    plot( xinfer[yinfer.ravel()==0, 0], xinfer[yinfer.ravel()==0, 1], '.', color=[0,0,.5], markersize=8, label='class 1' )
    plot( xinfer[yinfer.ravel()==1, 0], xinfer[yinfer.ravel()==1, 1], 'x', color=[0,0,.5], markersize=8, label='class 1 error' )
    xinferblue, yinferblue = xinfer, yinfer

    # plot red class
    xinfer = xtrain[ytrain.ravel()==1]
    yinfer = self.infer(xinfer)
    plot( xinfer[yinfer.ravel()==1, 0], xinfer[yinfer.ravel()==1, 1], '.', color=[.5,0,0], markersize=8, label='class 2' )
    plot( xinfer[yinfer.ravel()==0, 0], xinfer[yinfer.ravel()==0, 1], 'x', color=[.5,0,0], markersize=8, label='class 2 error' )
    xinferred, yinferred = xinfer, yinfer

    axis('image'); title(plttitle); legend(loc='lower left', framealpha=.5, fontsize=10); axis('off')

    # load data from surface plots
    if exists(join(logdir,'surface.pkl')):

      with open(join(logdir,'surface.pkl'), 'rb') as f:
        cfeed, xent, acc, spec = pickle.load(f)

      # surface of xent
      plt.subplot2grid((3,4), (0,0))
      plot(cfeed, xent, '-', color='orange')
      plot(cfeed[index], xent[index], 'ko', markersize=8)
      title('xent'); ylim(0, 20)
      plt.gca().axes.get_xaxis().set_ticklabels([])

      # surface of acc
      plt.subplot2grid((3,4), (1,0))
      plot(cfeed, acc, '-', color='green')
      plot(cfeed[index], acc[index], 'ko', markersize=8)
      title('acc'); ylim(0, 1.05)
      plt.gca().axes.get_xaxis().set_ticklabels([])

      # surface of spec
      plt.subplot2grid((3,4), (2,0))
      plot(cfeed, spec, '-', color='cyan')
      plot(cfeed[index], spec[index], 'ko', markersize=8)
      title('curv'); ylim(0, 1700000)

    suptitle(args.sugg); tight_layout()

    # image metadata and save image
    os.makedirs(join(logdir, 'images'), exist_ok=True)
    savefig(join(logdir, 'images', name))
    # if name=='plot.jpg': experiment.log_image(join(logdir, 'images/plot.jpg')); os.remove(join(logdir, 'images/plot.jpg'))
    sleep(.1)
    close('all')
    
    # save the data needed to reproduce plot
    if args.saveplotdata:
      with open(join(logdir, name[:-4]+'.pkl'), 'wb') as f:
        pickle.dump(dict(xinferred=xinferred, yinferred=yinferred, xinferblue=xinferblue, yinferblue=yinferblue,
                         xx1=xx1, xx2=xx2, yy=yy, xtrain=xtrain, ytrain=ytrain), f)
      

  def wiggle(self, xdata, ydata, span=1, along='random'):
    '''perturb weights and plot the decision boundary at each step, also get loss surface'''

    # produce random direction
    weights = self.sess.run(tf.trainable_variables())
    if along=='random':
      direction = utils.get_random_dir(self.sess, self.filtnorms, weights)
      direction[-2] = direction[-2][:, None] # a hack to make it work
    elif along=='eigvec':
      eigval, direction, _, _, _ = self.get_hessian(xdata, ydata)
      
    # name of the surface sweep for comet
    name = 'span_' + str(args.span) + '/' + basename(args.pretrain_dir) + '/' + along # name of experiment

    # coordinates to plot within span
    cfeed = span/2 * np.linspace(-1, 1, args.nspan) ** 5
    cfeed = np.concatenate([cfeed.copy(), np.flip(cfeed.copy(), axis=0), cfeed.copy()])
    cfeed = cfeed[args.nspan // 2:-args.nspan // 2 + 1]
    plt.plot(cfeed)
    plt.savefig('cfeed.jpg')

    # loop over all points along surface direction
    xent = np.zeros(len(cfeed))
    acc = np.zeros(len(cfeed))
    spec = np.zeros(len(cfeed))
    weights = self.sess.run(tf.trainable_variables())
    for i, c in enumerate(cfeed):

      # perturbe the weights
      perturbedWeights = [w + c * d for w, d in zip(weights, direction)]

      # visualize what happens to decision boundary when weights are wiggled
      self.assign_weights(perturbedWeights)
      if exists(join(logdir,'surface.pkl')): self.plot(xdata, ydata, name=str(i/1000)+'.jpg', plttitle=format(c,'.3f'), index=i)

      # compute the loss surface
      # xent[i], acc[i] = self.evaluate(xdata, ydata)
      spec[i], _, projvec_corr, acc[i], xent[i] = self.get_hessian(xtrain, ytrain)
      # experiment.log_metric('xent', xent[i], step=i)
      # experiment.log_metric('acc', acc[i], step=i)
      # experiment.log_metric('spec', spec[i], step=i)

      print('progress:', i + 1, 'of', len(cfeed), '| xent:', xent[i])

    # make gif or save surface
    if exists(join(logdir,'surface.pkl')): # make gif of the decision boundary plots
      gifname = args.sugg + '.gif'
      os.system('python make_gif.py '+join(logdir, 'images')+' '+join(logdir, gifname))
      # experiment.log_asset(join(logdir, gifname))
      # os.system('dbx upload '+join(logdir, gifname)+' ckpt/swissroll/'+args.sugg+'/')

    # save the surface data
    with open(join(logdir, 'surface.pkl'), 'wb') as f:
      pickle.dump((cfeed, xent, acc, spec), f)
      # experiment.log_asset(join(logdir, 'surface.pkl'))


  def rollout(self, xdata, ydata, span=1, seed=args.seed):
    '''continuously compute rollouts in random directions and log to comet for later analysis'''
    
    np.random.seed(seed)

    # coordinates to plot within span
    cfeed = span/2 * np.linspace(0, 1, args.nspan) ** 5

    # get unperturbed weights
    weights = self.sess.run(tf.trainable_variables())

    # continuously loop to get many rollouts
    xents = []
    for trial in range(1000):
      
      # produce random direction
      tic = time()
      direction = utils.get_random_dir(self.sess, self.filtnorms, weights)
      direction[-2] = direction[-2][:, None] # a hack to make it work

      # loop over all points along surface direction
      xent = []
      for i, c in enumerate(cfeed):
        
        # perturbe the weights
        perturbedWeights = [w + c * d for w, d in zip(weights, direction)]
      
        # visualize what happens to decision boundary when weights are wiggled
        self.assign_weights(perturbedWeights)
      
        # compute the loss surface
        xent.append(self.evaluate(xdata, ydata)[0])
      
      # gather data on how fast things are going
      xents.append(xent)
      
      if not trial % 100:
        ttrial = time() - tic
        print('trial ' + str(trial) + ' done, ttrial=' + str(ttrial))
        # experiment.log_metric('ttrial', ttrial, step=trial)
        # experiment.log_metric('trial', trial, step=trial)
        plt.semilogy(cfeed, xent, '.-')
        # experiment.log_figure()
        plt.clf()
      
    filename = join(logdir, '..', args.sugg + '.pkl')
    with open(filename, 'wb') as f:
      pickle.dump(xents, f)
  
  def curv(self, xdata, ydata):
    '''get curvature (via hessian) along a random direction'''
    
    specs = []
    for trial in range(15000):
      spec, _, projvec_corr, acc_clean, xent_clean = self.get_hessian(xdata, ydata)
      specs.append(spec)
      if not trial % 100:
        # experiment.log_metric('trial', trial, step=trial)
        print('trial ', trial)
    # save to file
    with open('curv.pkl', 'wb') as f:
      pickle.dump(specs, f)
    # experiment.log_asset('curv.pkl', file_name='curv.pkl')
  
  def assign_weights(self, weights):
    '''assign weights (list of numpy arrays) into tf'''
    self.sess.run(self.assign_op, {node: value for node, value in zip(self.inputweights, weights)})

  def save(self):
    '''save model'''
    ckpt_state = tf.train.get_checkpoint_state(logdir)
    ckpt_file = join(logdir, 'model.ckpt')
    print('Saving model to '+ckpt_file)
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(self.sess, ckpt_file)
    os.system('dbx upload '+logdir+' ckpt/swissroll/' + args.tag[1:] + '/')
    # experiment.log_asset_folder(logdir)

def spectral_radius(xent, regularizable, projvec_beta=.55):
  """returns principal eig of the hessian"""

  # create initial projection vector (randomly and normalized)
  projvec_init = [np.random.randn(*r.get_shape().as_list()) for r in regularizable]
  magnitude = np.sqrt(np.sum([np.sum(p**2) for p in projvec_init]))
  projvec_init = projvec_init/magnitude

  # projection vector tensor variable
  with tf.variable_scope('projvec'):
    projvec = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                               trainable=False, initializer=tf.constant_initializer(p))
               for r,p in zip(regularizable, projvec_init)]

  # layer norm
  # norm_values = utils.layernormdev(regularizable)
  norm_values = utils.filtnorm(regularizable)
  projvec_mul_normvalues = [tf.multiply(f,p) for f,p in zip(norm_values, projvec)]

  # get the hessian-vector product
  gradLoss = tf.gradients(xent, regularizable)
  hessVecProd = tf.gradients(gradLoss, regularizable, projvec_mul_normvalues)
  hessVecProd = [h*n for h,n in zip(hessVecProd, norm_values)]

  # principal eigenvalue: project hessian-vector product with that same vector
  xHx = utils.list2dotprod(projvec, hessVecProd)

  # comopute next projvec
  if args.curv:
    nextProjvec = [tf.random_normal(shape=p.get_shape()) for p in projvec]
  else:
    normHv = utils.list2norm(hessVecProd)
    unitHv = [tf.divide(h, normHv) for h in hessVecProd]
    nextProjvec = [tf.add(h, tf.multiply(p, projvec_beta)) for h,p in zip(unitHv, projvec)]
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]

  # diagnostics: dotprod and euclidean distance of new projection vector from previous
  projvec_corr = utils.list2dotprod(nextProjvec, projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([projvec_corr]):
    with tf.variable_scope('projvec_op'):
      projvec_op = [tf.assign(p,n) for p,n in zip(projvec, nextProjvec)]

  return xHx, projvec_op, projvec_corr, projvec_mul_normvalues

if __name__ == '__main__':
  
  tf.reset_default_graph()
  logdir = join('./ckpt/', args.sugg)
  os.makedirs(logdir, exist_ok=True)

  if any([a.find('nhidden1')!=-1 for a in sys.argv[1:]]):
    args.nhidden = [args.nhidden1, args.nhidden2, args.nhidden3, args.nhidden4, args.nhidden5, args.nhidden6]

  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  np.random.seed(args.seed)

  # make dataset
  X, y = twospirals(args.ndata//2, noise=args.noise)
  order = np.random.permutation(len(X))
  X = X[order]
  y = y[order]
  splitIdx = int(.5 * len(X))
  xtrain, ytrain = X[:splitIdx], y[:splitIdx, None]
  xtest, ytest = X[splitIdx:], y[splitIdx:, None]
  
  if args.batchsize==None: args.batchsize = len(xtrain); print('fullbatch gradient descent')

  # make model
  model = Model(args)
  if args.wiggle:
    model.wiggle(xtrain, ytrain, args.span, args.along)
  elif args.rollout:
    model.rollout(xtrain, ytrain, args.span)
  elif args.curv:
    model.curv(xtrain, ytrain)
  elif args.justplot:
    model.plot(xtrain, ytrain)
  else:
    model.fit(xtrain, ytrain, xtest, ytest)
    model.plot(xtrain, ytrain)
    if args.save: model.save()
  print('done!')