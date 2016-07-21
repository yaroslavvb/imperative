from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

try:
  from tensorflow.contrib.imperative.python.imperative import test_util
  import tensorflow.contrib.imperative as imperative
except:
  import imperative
  from imperative import test_util

def mnistCost(train_data_flat, train_labels, x0, env):
  """Creates a simple linear model that evaluates cross-entropy loss and
  gradient on MNIST dataset. Mirrors 'linear' model from train-on-mnist.lua

  Result is a Python callable that accepts ITensor parameter vector and
  returns ITensor loss and gradient. It works as a plug-in replacement of
  "opfunc" in train-on-mnist

  IE, you can do:
  x = ti.ones(...)
  opfunc=mnist_model(x0)
  loss, grad = opfunc(x0)
  x1 = lbfgs(opfunc,...)
  """

  batchSize = 100

  # create our input end-point, this is where ITensor->Tensor conversion
  # happens
  param = env.make_input(x0)

  # reshape flat parameter vector into W and b parameter matrices
  W_flat = tf.slice(param, [0], [10240])
  W = tf.reshape(W_flat, [1024, 10])
  b_flat = tf.slice(param, [10240], [10])
  b = tf.reshape(b_flat, [1, 10])

  # create model
  data = tf.Variable(tf.zeros_initializer((batchSize, 1024)))
  targets = tf.Variable(tf.zeros_initializer((batchSize, 10)))
  logits = tf.matmul(data, W) + b
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)

  # create loss and gradient ops
  cross_entropy_loss = tf.reduce_mean(cross_entropy)
  Wnorm = tf.reduce_sum(tf.square(W))
  bnorm = tf.reduce_sum(tf.square(b))
  loss = cross_entropy_loss + (bnorm + Wnorm)/2
  [grad] = tf.gradients(loss, [param])

  # initialize data and targets. Load entire dataset into tf Variable
  data_placeholder = tf.placeholder(dtype=tf.float32)
  data_init = data.assign(data_placeholder)
  labels_placeholder = tf.placeholder(shape=(batchSize), dtype=tf.int32)
  labels_onehot = tf.one_hot(labels_placeholder - 1, 10)
  targets_init = targets.assign(labels_onehot)
  env.sess.run(data_init, feed_dict={data_placeholder:
                                     train_data_flat[:batchSize]})
  env.sess.run(targets_init, feed_dict={labels_placeholder:
                                        train_labels[:batchSize]})

  # create imperative wrapper of tensorflow graph we just constructed
  # ITensor input is automatically converged and fed into param
  # and outputs are converted to ITensor objects and returned
  return env.make_function(inputs=[param], outputs=[loss, grad])


class LbfgsTest(tf.test.TestCase):

  def testLbfgsTraining(self):
    # create imperative environment
    env = imperative.Env(tf)
    ti = env.tf
    env.set_default_graph()  # set env's graph as default graph

    # Load 100 mnist training images/labels
    prefix = 'tensorflow/contrib/imperative/python/imperative/testdata'
    if not os.path.exists(prefix):
      prefix = os.path.dirname(os.path.realpath(__file__))+"/testdata"

    data_filename = prefix+"/mnist_data_32x32_small.npy"
    labels_filename = prefix+"/mnist_labels_small.npy"


    # work-around for Jenkins setup issue
    # https://github.com/tensorflow/tensorflow/issues/2855
    if not (os.path.exists(data_filename) and
            os.path.exists(labels_filename)):
      print("Couldn't find data dependency, aborting.")
      return True

    train_data = np.load(data_filename).reshape((-1, 1024))
    train_labels = np.load(labels_filename)

    x0 = ti.ones((10250))
    opfunc = mnistCost(train_data, train_labels, x0, env)

    def verbose_func(s):
      print(s)
  
    def dot(a, b):
      """Dot product function since TensorFlow doesn't have one."""
      return ti.reduce_sum(a*b)

    def lbfgs(opfunc, x, config, state):
      """Line-by-line port of lbfgs.lua, using TensorFlow imperative mode.
      """

      maxIter = config.maxIter or 20
      maxEval = config.maxEval or maxIter*1.25
      tolFun = config.tolFun or 1e-5
      tolX = config.tolX or 1e-9
      nCorrection = config.nCorrection or 100
      lineSearch = config.lineSearch
      lineSearchOpts = config.lineSearchOptions
      learningRate = config.learningRate or 1
      isverbose = config.verbose or False

      # verbose function
      if isverbose:
        verbose = verbose_func
      else:
        verbose = lambda x: None

      # evaluate initial f(x) and df/dx
      f, g = opfunc(x)

      f_hist = [f]
      currentFuncEval = 1
      state.funcEval = state.funcEval + 1
      p = g.shape[0]

      # check optimality of initial point
      tmp1 = ti.abs(g)
      if ti.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist

      # optimize for a max of maxIter iterations
      nIter = 0
      while nIter < maxIter:
        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1

        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
          d = -g
          old_dirs = []
          old_stps = []
          Hdiag = 1
        else:
          # do lbfgs update (update memory)
          y = g - g_old
          s = d*t
          ys = dot(y, s)

          if ys > 1e-10:
            # updating memory
            if len(old_dirs) == nCorrection:
              # shift history by one (limited-memory)
              del old_dirs[0]
              del old_stps[0]

            # store new direction/step
            old_dirs.append(s)
            old_stps.append(y)

            # update scale of initial Hessian approximation
            Hdiag = ys/dot(y, y)

          # compute the approximate (L-BFGS) inverse Hessian 
          # multiplied by the gradient
          k = len(old_dirs)

          # need to be accessed element-by-element, so don't re-type tensor:
          ro = [0]*nCorrection
          for i in range(k):
            ro[i] = 1/dot(old_stps[i], old_dirs[i])


          # iteration in L-BFGS loop collapsed to use just one buffer
          # need to be accessed element-by-element, so don't re-type tensor:
          al = [0]*nCorrection

          q = -g
          for i in range(k-1, -1, -1):
            al[i] = dot(old_dirs[i], q) * ro[i]
            q = q - al[i]*old_stps[i]

          # multiply by initial Hessian
          r = q*Hdiag
          for i in range(k):
            be_i = dot(old_stps[i], r) * ro[i]
            r += (al[i]-be_i)*old_dirs[i]

          d = r
          # final direction is in r/d (same object)

        g_old = g
        f_old = f

        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = dot(g, d)

        # check that progress can be made along that direction
        if gtd > -tolX:
          verbose("Can not make progress along direction.")
          break

        # reset initial guess for step size
        if state.nIter == 1:
          tmp1 = ti.abs(g)
          t = min(1, 1/ti.reduce_sum(tmp1))
        else:
          t = learningRate


        # optional line search: user function
        lsFuncEval = 0
        if lineSearch and isinstance(lineSearch) == types.FunctionType:
          # perform line search, using user function
          f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
          f_hist.append(f)
        else:
          # no line search, simply move with fixed-step
          x += t*d

          if nIter != maxIter:
            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            f, g = opfunc(x)

            lsFuncEval = 1
            f_hist.append(f)


        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
          # no use to run tests
          verbose('reached max number of iterations')
          break

        if currentFuncEval >= maxEval:
          # max nb of function evals
          verbose('max nb of function evals')
          break

        tmp1 = ti.abs(g)
        if ti.reduce_sum(tmp1) <=tolFun:
          # check optimality
          verbose('optimality condition below tolFun')
          break

        tmp1 = ti.abs(d*t)
        if ti.reduce_sum(tmp1) <= tolX:
          # step size below tolX
          verbose('step size below tolX')
          break

        if ti.abs(f-f_old) < tolX:
          # function value changing less than tolX
          verbose('function value changing less than tolX'+str(ti.abs(f-f_old)))
          break


      # save state
      state.old_dirs = old_dirs
      state.old_stps = old_stps
      state.Hdiag = Hdiag
      state.g_old = g_old
      state.f_old = f_old
      state.t = t
      state.d = d

      return x, f_hist, currentFuncEval

    # initialize l-BFSG parameters
    state = Struct()
    config = Struct()
    config.nCorrection = 5
    config.maxIter = 30
    config.verbose = True

    x, f_hist, currentFuncEval = lbfgs(opfunc, x0, config, state)
    self.assertTrue(f_hist[0]>5000)
    self.assertTrue(f_hist[29]<200)

    
# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

if __name__ == "__main__":
  tf.test.main()

