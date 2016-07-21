"""Implementation of Imperative  environment. All user-facing elements
of framework should go here.

Env: imperative environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np

from .itensor import ITensor
from .module_rewriter import ImperativeRewriter
from .module_rewriter import ModuleRewriter
from .op import Op
from .util import get_current_device_string
from .util import is_list_or_tuple
from .util import shorten_device_string

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import gen_math_ops


# x-version compatibility
# constant_op moved in a558c6e3b38846727873b5afbbc3ba309ae5dff5
try:
  from tensorflow.python.framework import constant_op
except ImportError:
  from tensorflow.python.ops import constant_op



__all__ = ["Env"]

# Global env to be reused between threads.
_global_default_env = None

class Env(object):
  """Env is an object that manages current graph and session and translates
  user commands into appropriate session.run calls.

  The translation is done by means of wrapping all of tf functions that accept
  Tensor objects into a version that accepts ITensor objects that represent
  tensors with a concrete value.

  It keeps track of operations added to the current graph and tried to reuse
  parts of existing graph when possible.

  import tensorflow as tf
  env = imperative.Env(tf)
  c = env.tf.add(1, 2)

  """

  def __init__(self, tf_namespace, config=None):
    """Creates new imperative environment.

    Args:
      tf_namespace: tensorflow namespace to wrap, or a dictionary of namespace
          name-namespace pairs to wrap multiple namespaces
      config: ConfigProto to use for configuring session
    """

    global _global_default_env

    self.CACHE_ENABLED = True
    self.PRINT_CACHE_MISSES = False
    self.PRINT_CACHE_HITS = False

    self.op_cache = {}  # cache used for reusing parts of graph
    self.g = ops_lib.Graph()
    self.sess = session.Session(config=config, graph=self.g)
    self._gc_default = self.session._DEAD_HANDLES_THRESHOLD

    # wrap provided namespace for imperative mode
    symbol_rewriter = ImperativeRewriter(self)
    rewriter = ModuleRewriter(symbol_rewriter, "imperative.")

    # tf_namespace is like {"tf": tf, "gen_math_ops": gen_math_ops}
    if isinstance(tf_namespace, dict):
      assert "tf" in tf_namespace, "Must include main tf namespace."
      self.original_tf = tf_namespace["tf"]
      for name, namespace in tf_namespace.items():
        self.__dict__[name] = rewriter(namespace)
    else: # tf_namespace is like "tf"
      self.original_tf = tf_namespace
      self.tf = rewriter(tf_namespace)

    # fields used when recording tracing runs
    self._tracing_enabled = False
    self._tracing_options = self.tf.RunOptions(trace_level=
                                               self.tf.RunOptions.FULL_TRACE)
    self._run_metadata = self.tf.RunMetadata()

    # used for creating functions
    self.input_dict = {}

    self.default_session = None
    self.default_graph = None

    _global_default_env = self


  def disable_gc(self):
    """Turn off garbage collection for persistent Tensors."""
    # it's saved in abstract base class of Session
    self.session.__class__.__base__._DEAD_HANDLES_THRESHOLD = 10**20



  def enable_gc(self):
    """Turn on garbage collection for persistent Tensors."""
    self.session.__class__.__base__._DEAD_HANDLES_THRESHOLD = self._gc_default


  def enable_tracing(self):
    self._tracing_enabled = True

  def disable_tracing(self):
    self._tracing_enabled = False

  @staticmethod
  def get_global_default_env():
    """Get global env for reuse (ie, in tests)."""
    return _global_default_env


  def close(self):
    """Close Env and free its resources."""
    self.sess.close()


  @property
  def session(self):
    """Session associated with current imperative env."""
    return self.sess


  @property
  def graph(self):
    """Graph associated with current imperative env."""
    return self.g


  @property
  def _graph_version(self):
    """Gives version of the graph. This can be used for checking if graph
    modifications took place"""
    return self.g.version


  @property
  def device(self):
    """Context manager for the default device to use in this env."""
    return self.g.device


  def cache_lookup(self, key):
    """Retrieve Op object from the cache."""
    if self.CACHE_ENABLED:
      if self.PRINT_CACHE_HITS:
        print("Imperative cache hit for %s" %(str(key)))
      return self.op_cache.get(key, None)


  def cache_add(self, key, op):
    """Add given Op object to the cache."""
    self.op_cache[key] = op


  def handle_to_numpy(self, tensor_handle):
    """Download contents of TensorHandle and return corresponding numpy array.

    Args:
      tensor_handle: session_ops.TensorHandle object

    Returns:
      numpy array with a copy of data from tensor_handle
    """

    tf_dtype = tensor_handle._dtype
    current_device = get_current_device_string(self.g)
    current_device_sanitized = current_device.replace(":", "")

    device_func = session_ops.TensorHandle._get_device_name
    handle_device = device_func(tensor_handle.handle)
    handle_device = shorten_device_string(handle_device)
    handle_device_sanitized = handle_device.replace(":", "")

    key = ("handle_to_numpy", tf_dtype.name, handle_device, current_device)

    if key in self.op_cache:
      holder, tensor = self.op_cache[key]
    else:
      if self.PRINT_CACHE_MISSES:
        print("Imperative cache miss for %s"%(str(key)))

      op_prefix = "handle_to_numpy.%s.%s.%s" % (tf_dtype.name,
                                                handle_device_sanitized,
                                                current_device_sanitized)
      with self.g.as_default():
        holder, tensor = session_ops.get_session_tensor(tensor_handle.handle,
                                                        tensor_handle._dtype,
                                                        name=op_prefix)
      self.op_cache[key] = (holder, tensor)

    return self.run(tensor, feed_dict={holder: tensor_handle.handle})


  def numpy_to_handle(self, array):
    """Upload numpy array into TensorFlow runtime.

    Args:
      array: numpy array to convert to TensorHandle

    Returns:
      TensorHandle corresponding to given numpy array.
    """

    tf_dtype = dtypes.as_dtype(array.dtype)
    current_device = get_current_device_string(self.g)
    current_device_sanitized = current_device.replace(":", "")
    key = ("numpy_to_handle", tf_dtype.name, current_device)

    if key in self.op_cache:
      holder, handle_op = self.op_cache[key]
    else:
      if self.PRINT_CACHE_MISSES:
        print("Imperative cache miss for %s"%(str(key)))

      op_prefix = "numpy_to_handle.%s.%s" % (tf_dtype.name,
                                             current_device_sanitized)
      with self.g.as_default():
        holder = array_ops.placeholder(dtype=array.dtype,
                                       name=op_prefix+".holder")
        handle_op = session_ops.get_session_handle(holder,
                                                   name=op_prefix+".handle")
      self.op_cache[key] = (holder, handle_op)

    handle = self.run(handle_op, feed_dict={holder: array})
    return handle


  def handle_to_itensor(self, handle):
    return ITensor(self, handle)

  def itensor_to_numpy(self, itensor):
    """Convert itensor to numpy array."""

    if itensor.env != self:
      raise ValueError("ITensor has incompatible env")
    return itensor.as_numpy()


  def numpy_to_itensor(self, array, dtype=None, shape=None):
    """Convert numpy.ndarray or compatible type to imperative.Tensor."""

    # convert to numpy dtype if necessary
    if dtype:
      tf_dtype = dtypes.as_dtype(dtype)
      np_dtype = tf_dtype.as_numpy_dtype
    else:
      np_dtype = None

    if isinstance(array, ITensor):
      raise ValueError("Passed ITensor instead of numpy into "
                       "numpy_to_itensor.")

    # try to convert Python lists to numpy array
    if not isinstance(array, np.ndarray):
      array = np.array(array, dtype=np_dtype)
      tf_dtype = dtypes.as_dtype(array.dtype)

      if not tf_dtype or array.dtype == np.dtype("O"):
        raise ValueError("Unsupported type %s" %(type(array)))

    # Follow downcasting convention as in python/framework/tensor_util.py#L357
    # python/numpy default float type is float64. We prefer float32 instead.
    if (array.dtype == np.float64) and dtype is None:
      array = array.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif (array.dtype == np.int64) and dtype is None:
      downcasted_array = array.astype(np.int32)
      # Do not down cast if it leads to precision loss.
      if np.array_equal(downcasted_array, array):
        array = downcasted_array

    # if dtype is not None, and doesn't match given ndarray, convert to that
    # type
    if np_dtype and array.dtype != np_dtype:
      array = array.astype(np_dtype)

    if shape and array.shape != shape:
      array = array.reshape(shape)

    handle = self.numpy_to_handle(array)
    return ITensor(self, handle)

  def tensor_to_itensor(self, tensor):

    op_prefix = "tensor_to_itensor"
    with self.g.as_default():
      handle_op = session_ops.get_session_handle(tensor,
                                                 name=op_prefix+".handle")
    handle = self.run(handle_op)
    return ITensor(self, handle)


  def constant(self, values, dtype=None, shape=None, name="Const"):
    """Imperative specific implementation of constant-op."""

    np_dtype = None

    # Convert numpy dtype to TensorFlow dtype if needed
    if dtype:
      try:
        dtype = dtypes.as_dtype(dtype)
        np_dtype = dtype.as_numpy_dtype
      except TypeError as exc:
        raise TypeError("Trying to create constant with dtype=%s, "
                        "got TypeError(%s)" % (dtype, exc.message))

    # Native TensorFlow has special handling for TensorProto initialized with
    # a scalar and non-empty shape. For feature parity with TensorFlow we
    # handle this case by tiling the constant explicitly.
    if isinstance(values, numbers.Number) and shape:
      data_array = values*np.ones(shape=shape, dtype=np_dtype)
      return self.numpy_to_itensor(data_array,
                                   dtype=dtype, shape=shape)

    return self.numpy_to_itensor(values, dtype, shape)


  def make_input(self, x, name=""):
    """Returns Tensor of the same type/device as x which can be used
    as input to native TensorFlow ops, and substituted later with an ITensor,
    using callable created with env.make_function(). The user must ensure
    that future ITensor is on the same device as x, otherwise you will see
    memcpy/CUDA sync errors.

    Args:
      x: ITensor used to initalize input tensor. It used only to determine
          dtype and device placement.

    Returns:
      A Tensor that can be used in TensorFlow ops.
    """
    op_name = "custom_input_%s"%(name)
    input_holder, input_ = session_ops.get_session_tensor(x.tf_handle,
                                                          x.dtype,
                                                          name=op_name)

    self.input_dict[input_] = input_holder
    return input_


  def make_function(self, inputs, outputs, name=""):
    """Create callable that accept argument ITensors in the same order as
    inputs argument, and produces tuple of outputs which are ITensors
    corresponding to outputs.

    Example usage:
    x0 = env.tf.ones()       # create ITensor
    x = env.make_input(x0) # create Tensor
    y = env.make_input(x0) # create Tensor
    z1 = tf.add(x, y)         # operate on Tensors
    z2 = tf.sub(x, y)         # operate on Tensors
    f = env.make_function(inputs=[x, y], outputs=[z1, z2])

    print(f(x0, x0*5))       # feed ITensors, get result back as ITensors
    """

    input_holders = []
    for input_ in inputs:
      input_holders.append(self.input_dict[input_])

    output_handle_ops = []
    if is_list_or_tuple(outputs):
      for (i,tensor) in enumerate(outputs):
        op_name = "custom_function_%s.output.%s"%(name, i)
        output_handle_ops.append(session_ops.get_session_handle(tensor,
                                                                op_name))
    # special-case single output
    else:
      op_name = "custom_function_%s.output"%(name)
      output_handle_ops = session_ops.get_session_handle(outputs, op_name)

    def func(*args):
      feed_dict = {}
      for (i, arg) in enumerate(args):
        feed_dict[input_holders[i]] = arg.tf_handle

      tensor_handles = self.sess.run(output_handle_ops, feed_dict=feed_dict)
      if is_list_or_tuple(tensor_handles):
        return [ITensor(self, t) for t in tensor_handles]
      else:
        return ITensor(self, tensor_handles)
      
    return func


  # faster version for summing over flat tensors (5x faster than using
  # native with unknown size)
  # TODO(yaroslavvb): respect is_cache_enabled.
  # TODO(yaroslavvb): deprecate since make_function is more elegant solution?
  def sum1(self, input_itensor):
    """Create a specialized op that sums over 1 dimensional vector.
    This avoids having to create Rank/Range ops that initialize indices
    in the default tf.reduce_sum."""

    op_type_name = "sum1"
    tf_dtype = input_itensor.dtype
    current_device = get_current_device_string(self.g)
    current_device_sanitized = current_device.replace(":", "")
    key = (op_type_name, tf_dtype.name, current_device_sanitized)

    if key in self.op_cache:
      if self.PRINT_CACHE_HITS:
        print("Imperative cache hit for %s"%(str(key)))
      op = self.op_cache[key]
    else:
      if self.PRINT_CACHE_MISSES:
        print("Imperative cache miss for %s"%(str(key)))
      with self.g.as_default():
        op_prefix = op_type_name + "." + tf_dtype.name
        holder, tensor = session_ops.get_session_tensor(
            input_itensor.tf_handle, input_itensor.dtype, name=op_prefix+".0")
        input_holders = {"input": holder}
        reduction_indices = constant_op.constant([0], dtype=dtypes.int32,
                                                 name=op_prefix+".1")
        output = gen_math_ops._sum(input=tensor,
                                   reduction_indices=reduction_indices,
                                   keep_dims=False, name=op_prefix+".op")
        op_prefix = op_prefix+".out"
        output_handle = session_ops.get_session_handle(output,
                                                       op_prefix+".handle")

      op = Op(self, input_holders, output_handle)
      self.cache_add(key, op)

    return op(input=input_itensor)


  def set_default_graph(self):
    """Sets default graph to the graph of this imperative environment."""
    self.default_graph = self.g.as_default()
    self.default_graph.enforce_nesting = False
    self.default_graph.__enter__()


  def set_default_session(self):
    """Sets default graph to the graph of this imperative environment."""
    self.default_session = self.sess.as_default()
    self.default_session.enforce_nesting = False
    self.default_session.__enter__()


  # TODO(yaroslavvb): rename into _run because it's implementation internal
  def run(self, *args, **kwargs):
    """Execute session.run in the current Env."""
    if self._tracing_enabled:
      kwargs["options"] = self._tracing_options
      kwargs["run_metadata"] = self._run_metadata
      return self.sess.run(*args, **kwargs)
    else:
      return self.sess.run(*args, **kwargs)
