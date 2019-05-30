import tensorflow as tf
import numpy as np
import utils
import time

np.random.seed(1234)

def _spec(net, xentPerExample, is_accum=False, nohess=False, randvec=False):
  """returns principal eig of the hessian"""

  if nohess:
    net.valtotEager = net.bzEager = net.valEager = net.valtotAccum = net.bzAccum = net.valAccum = tf.constant(0, tf.float32)
    net.projvec = net.projvec_op = net.projvec_corr = tf.constant(0, tf.float32)
    return

  batchsize = tf.shape(xentPerExample)[0]
  xent = tf.reduce_sum(xentPerExample)

  # decide weights from which to compute the spectral radius
  print('Number of trainable weights: ' + str(utils.count_params(tf.trainable_variables())))
  if not net.args.specreg_bn: # don't include batch norm weights
    net.regularizable = []
    for var in tf.trainable_variables():
      if var.op.name.find('logit/dense/kernel') > -1 or var.op.name.find(r'DW') > -1:
        net.regularizable.append(var)
    print('Number of regularizable weights: ' + str(utils.count_params(net.regularizable)))
  else:
    net.regularizable = tf.trainable_variables() # do include bn weights
    print('Still zeroing out bias and bn variables in hessian calculation in utils.filtnorm function')

  # create initial projection vector (randomly and normalized)
  projvec_init = [np.random.randn(*r.get_shape().as_list()) for r in net.regularizable]
  magnitude = np.sqrt(np.sum([np.sum(p**2) for p in projvec_init]))
  projvec_init = [p/magnitude for p in projvec_init]

  # projection vector tensor variable
  net.count = net.count + 1 if hasattr(net, 'count') else 0
  with tf.variable_scope('projvec/'+str(net.count)):
    net.projvec = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                                   trainable=False, initializer=tf.constant_initializer(p))
                   for r,p in zip(net.regularizable, projvec_init)]

  # compute filter normalization
  print('normalization scheme: '+net.args.normalizer)
  if net.args.normalizer == None or net.args.normalizer=='None':
    projvec_mul_normvalues = net.projvec
  else:
    if net.args.normalizer == 'filtnorm': normalizer = utils.filtnorm
    elif net.args.normalizer == 'layernorm': normalizer = utils.layernorm
    elif net.args.normalizer == 'layernormdev': normalizer = utils.layernormdev
    net.normvalues = normalizer(net.regularizable)
    projvec_mul_normvalues = [n*p for n,p in zip(net.normvalues, net.projvec)]

  # get gradient of loss wrt inputs
  tstart = time.time(); gradLoss = tf.gradients(xent, net.regularizable); print('Built gradLoss: ' + str(time.time() - tstart) + ' s')

  # get hessian vector product
  tstart = time.time()
  hessVecProd = tf.gradients(gradLoss, net.regularizable, projvec_mul_normvalues)
  # hessVecProd = [h*n for h,n in zip(hessVecProd, net.normvalues)]
  print('Built hessVecProd: ' + str(time.time() - tstart) + ' s')

  # build graph for full-batch hessian calculations which require accum ops and storage variables (for validation)
  if is_accum:

    # create op to accumulate gradients
    with tf.variable_scope('accum'):
      hessvecprodAccum = [tf.Variable(tf.zeros_like(h), trainable=False, name=h.op.name) for h in hessVecProd]
      batchsizeAccum = tf.Variable(0, trainable=False, name='batchsizeAccum')
      net.zero_op = [a.assign(tf.zeros_like(a)) for a in hessvecprodAccum] + [batchsizeAccum.assign(0)]
      net.accum_op = [a.assign_add(g) for a,g in zip(hessvecprodAccum, hessVecProd)] + [batchsizeAccum.assign_add(batchsize)]

    # compute the projected projection vector using accumulated hvps
    nextProjvec = compute_nextProjvec(net.projvec, hessvecprodAccum, net.projvec_beta, randvec=randvec)
    print('nextProjvec using accumed hvp')

    # hooks for total eigenvalue, batch size, and eigenvalue
    net.valtotAccum = utils.list2dotprod(net.projvec, hessvecprodAccum)
    net.bzAccum = tf.to_float(batchsizeAccum)
    net.valAccum = net.valtotAccum / net.bzAccum

  # build graph for on-the-fly per-batch hessian calcuations (for training)
  else:

    # compute the projected projection vector using instantaneous hvp
    nextProjvec = compute_nextProjvec(net.projvec, hessVecProd, net.projvec_beta, randvec=randvec)
    print('nextProjvec using instant hvp and randvec is', randvec)

    # hooks for total eigenvalue, batch size, and eigenvalue
    net.valtotEager = utils.list2dotprod(net.projvec, hessVecProd)
    net.bzEager = tf.to_float(batchsize)
    net.valEager = net.valtotEager / net.bzEager

  # dotprod and euclidean distance of new projection vector from previous
  net.projvec_corr = utils.list2dotprod(nextProjvec, net.projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([net.projvec_corr]):
    with tf.variable_scope('projvec_op'):
      net.projvec_op = [tf.assign(p,n) for p,n in zip(net.projvec, nextProjvec)]

def compute_nextProjvec(projvec, hvp, projvec_beta, randvec=False):
  '''get the next projvec'''

  # make the vector
  if randvec:
    nextProjvec = [tf.random_normal(shape=p.get_shape()) for p in projvec]
  else:
    normHv = utils.list2norm(hvp)
    unitHv = [tf.divide(h, normHv) for h in hvp]
    nextProjvec = [tf.add(h, tf.multiply(p, projvec_beta)) for h,p in zip(unitHv, projvec)]

  # normalize the vector
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]
  return nextProjvec


