import tensorflow as tf
import os
from os.path import join, exists
import argparse
from time import time, sleep
import numpy as np
import utils, resnet_model
from utils import timenow, Scheduler, Accumulator
import sys
from dataloaders_torch import get_loader
from resnet_evaluator import Evaluator
import subprocess
from shutil import rmtree

parser = argparse.ArgumentParser()
# options
parser.add_argument('-gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES=?')
parser.add_argument('-gpu_eval', action='store_true', help='use gpu in validation process, otherwise cpu')
parser.add_argument('-mode', default='train', type=str, help='train, or eval.')
parser.add_argument('-resume', action='store_true', help='use this to resume training from current checkpoint')
# various debugging flags. Please follow instructions on README if you don't know what's going on here
parser.add_argument('-poison', action='store_true')
parser.add_argument('-nogan', action='store_true')
parser.add_argument('-cinic', action='store_true')
parser.add_argument('-svhn', action='store_true')
parser.add_argument('-tanti', action='store_true')
parser.add_argument('-sigopt', action='store_true')
parser.add_argument('-nohess', action='store_true')
parser.add_argument('-randvec', action='store_true')
parser.add_argument('-noaugment', action='store_true')
parser.add_argument('-upload', action='store_true')
parser.add_argument('-randname', action='store_true')
# file names
parser.add_argument('-log_root', default='debug', type=str, help='Directory to keep the checkpoints.')
parser.add_argument('-tag', default=None, type=str, help='Project tag')
parser.add_argument('-ckpt_root', default='/root/ckpt', type=str, help='ckpt: directory of where checkpoints are stored')
parser.add_argument('-bin_path', default='/root/bin', type=str, help='bin: directory of helpful scripts')
parser.add_argument('-cifar100', action='store_true')
# network parameters
parser.add_argument('-num_resunits', default=3, type=int, help='Number of residual units n. There are 6*n+2 layers')
parser.add_argument('-resnet_width', default=1, type=int, help='Multiplier of the width of hidden layers. Base is (16,32,64)')
# training hyperparam
parser.add_argument('-lrn_rate', default=1e-1, type=float, help='initial learning rate to use for training')
parser.add_argument('-batch_size', default=128, type=int, help='batch size to use for training')
parser.add_argument('-weight_decay', default=0.0002, type=float, help='coefficient for the weight decay')
parser.add_argument('-epoch_end', default=256, type=int, help='ending epoch')
parser.add_argument('-max_grad_norm', default=25, type=float, help='maximum allowed gradient norm for hessian term')
# poison data
parser.add_argument('-nodirty', action='store_true')
parser.add_argument('-fracdirty', default=.95, type=float) # should be < .5 for now
# hessian regularization
parser.add_argument('-speccoef', default=1e-1, type=float, help='coefficient for the spectral radius')
parser.add_argument('-speccoef_init', default=0.0, type=float, help='pre-warmup coefficient for the spectral radius')
parser.add_argument('-warmupPeriod', default=20, type=int)
parser.add_argument('-specreg_bn', default=False, type=bool, help='include bn weights in the calculation of the spectral regularization loss?')
parser.add_argument('-normalizer', default='layernormdev', type=str, help='normalizer to use (filtnorm, layernorm, layernormdev)')
parser.add_argument('-projvec_beta', default=.5, type=float, help='discounting factor or "momentum" coefficient for averaging of projection vector')
# sharp hess
parser.add_argument('-n_grads_spec', default=2, type=int)
parser.add_argument('-specexp', default=12, type=float, help='exponent for spectral radius loss')
# load pretrained
parser.add_argument('-pretrain_url', default=None, type=str, help='url of pretrain directory')
parser.add_argument('-pretrain_dir', default=None, type=str, help='remote directory on dropbox of pretrain')

def train():

  # start evaluation process
  popen_args = dict(shell=True, universal_newlines=True, encoding='utf-8') # , stdout=PIPE, stderr=STDOUT, )
  command_valid = 'python main.py -mode=eval ' + ' '.join(['-log_root='+args.log_root] + sys.argv[1:])
  valid = subprocess.Popen(command_valid, **popen_args)
  print('EVAL: started validation from train process using command:', command_valid)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # eval may or may not be on gpu

  # build graph, dataloader
  cleanloader, dirtyloader, _ = get_loader(join(home, 'datasets'), batchsize=args.batch_size, poison=args.poison, svhn=args.svhn,
                                           fracdirty=args.fracdirty, cifar100=args.cifar100, noaugment=args.noaugment,
                                           nogan=args.nogan, cinic=args.cinic, tanti=args.tanti)
  dirtyloader = utils.itercycle(dirtyloader)
  model = resnet_model.ResNet(args, args.mode)

  # initialize session
  print('===================> TRAIN: STARTING SESSION at '+timenow())
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
  print('===================> TRAIN: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

  # load checkpoint
  utils.download_pretrained(log_dir, pretrain_dir=args.pretrain_dir) # download pretrained model
  ckpt_file = join(log_dir, 'model.ckpt')
  ckpt_state = tf.train.get_checkpoint_state(log_dir)
  var_list = list(set(tf.global_variables())-set(tf.global_variables('accum'))-set(tf.global_variables('projvec')))
  saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
  sess.run(tf.global_variables_initializer())
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    print('TRAIN: No pretrained model. Initialized from random')
  else:
    print('TRAIN: Loading checkpoint %s', ckpt_state.model_checkpoint_path)

  print('TRAIN: Start')
  scheduler = Scheduler(args)
  for epoch in range(args.epoch_end): # loop over epochs
    accumulator = Accumulator()

    # loop over batches
    for batchid, (cleanimages, cleantarget) in enumerate(cleanloader):

      # pull anti-training samples
      dirtyimages, dirtytarget = dirtyloader.__next__()

      # convert from torch format to numpy onehot, batch them, and apply softmax hack
      cleanimages, cleantarget, dirtyimages, dirtytarget, batchimages, batchtarget, dirtyOne, dirtyNeg = \
        utils.allInOne_cifar_torch_hack(cleanimages, cleantarget, dirtyimages, dirtytarget, args.nodirty, args.num_classes, args.nogan)

      # run the graph
      _, global_step, loss, predictions, acc, xent, xentPerExample, weight_norm = sess.run(
        [model.train_op, model.global_step, model.loss, model.predictions, model.precision, model.xent, model.xentPerExample, model.weight_norm],
        feed_dict={model.lrn_rate: scheduler._lrn_rate,
                   model._images: batchimages,
                   model.labels: batchtarget,
                   model.dirtyOne: dirtyOne,
                   model.dirtyNeg: dirtyNeg})

      metrics = {}
      metrics['clean/xent'], metrics['dirty/xent'], metrics['clean/acc'], metrics['dirty/acc'] = \
        accumulator.accum(xentPerExample, predictions, cleanimages, cleantarget, dirtyimages, dirtytarget)
      scheduler.after_run(global_step, len(cleanloader))

      if np.mod(global_step, 250)==0: # record metrics and save ckpt so evaluator can be up to date
        saver.save(sess, ckpt_file)
        metrics['lr'], metrics['train/loss'], metrics['train/acc'], metrics['train/xent'] = \
          scheduler._lrn_rate, loss, acc, xent
        metrics['clean_minus_dirty'] = metrics['clean/acc'] - metrics['dirty/acc']
        if 'timeold' in locals(): metrics['time_per_step'] = (time()-timeold)/250
        timeold = time()
        # experiment.log_metrics(metrics, step=global_step)
        print('TRAIN: loss: %.3f, acc: %.3f, global_step: %d, epoch: %d, time: %s' % (loss, acc, global_step, epoch, timenow()))

    # log clean and dirty accuracy over entire batch
    metrics = {}
    metrics['clean/acc_full'], metrics['dirty/acc_full'], metrics['clean_minus_dirty_full'], metrics['clean/xent_full'], metrics['dirty/xent_full'] = \
      accumulator.flush()
    # experiment.log_metrics(metrics, step=global_step)
    # experiment.log_metric('weight_norm', weight_norm)
    print('TRAIN: epoch', epoch, 'finished. cleanacc', metrics['clean/acc_full'], 'dirtyacc', metrics['dirty/acc_full'])

def evaluate():

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if not args.gpu_eval else args.gpu # run eval on cpu
  cleanloader, _, testloader = get_loader(join(home, 'datasets'), batchsize=args.batch_size, fracdirty=args.fracdirty,
                                          cifar100=args.cifar100, cinic=args.cinic, svhn=args.svhn, nworker=0)

  print('===================> EVAL: STARTING SESSION at '+timenow())
  evaluator = Evaluator(testloader, args)
  print('===================> EVAL: SESSION STARTED at '+timenow()+' on CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'])

  # continuously evaluate until process is killed
  best_acc = worst_acc = 0.0
  while True:
    metrics = {}

    # restore weights from file
    restoreError = evaluator.restore_weights(log_dir)
    if restoreError: print('no weights to restore'); sleep(1); continue

    # KEY LINE OF CODE
    xent, acc, global_step = evaluator.eval()
    best_acc = max(acc, best_acc)
    worst_acc = min(acc, worst_acc)

    # evaluate hessian as well
    val = corr_iter = corr_period = 0
    if not args.nohess:
      val, nextProjvec, corr_iter = evaluator.get_hessian(loader=cleanloader, num_power_iter=1, num_classes=args.num_classes)
      if 'projvec' in locals(): # compute correlation between projvec of different epochs
        corr_period = np.sum([np.dot(p.ravel(),n.ravel()) for p,n in zip(projvec, nextProjvec)]) # correlation of projvec of consecutive periods (5000 batches)
        metrics['hess/projvec_corr_period'] = corr_period
      projvec = nextProjvec

    # log metrics
    metrics['eval/acc'] = acc
    metrics['eval/xent'] = xent
    metrics['eval/best_acc'] = best_acc
    metrics['eval/worst_acc'] = worst_acc
    metrics['hess/val'] = val
    metrics['hess/projvec_corr_iter'] = corr_iter
    # experiment.log_metrics(metrics, step=global_step)
    print('EVAL: loss: %.3f, acc: %.3f, best_acc: %.3f, val: %.3f, corr_iter: %.3f, corr_period: %.3f, global_step: %s, time: %s' %
          (xent, acc, best_acc, val, corr_iter, corr_period, global_step, timenow()))


if __name__ == '__main__':

  # parse arg
  args = parser.parse_args()

  # programmatically modify args based on other args
  if args.resume: args.pretrain_dir = args.pretrain_url = None # dont load pretrained if resuming
  args.num_classes = 100 if args.cifar100 else 10

  # make log directory
  home = os.environ['HOME']
  log_dir = join(args.ckpt_root, 'poisoncifar', args.log_root)
  if not args.resume and args.mode=='train' and exists(log_dir): rmtree(log_dir)
  os.makedirs(log_dir, exist_ok=True)
  print('log_root: '+args.log_root)

  # start train/eval
  if args.mode == 'train':
    train()
    print('TRAIN: ALL DONE')

  elif args.mode == 'eval':
    evaluate()

