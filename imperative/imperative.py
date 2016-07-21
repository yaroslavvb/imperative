"""Amalgamation of imperative package, made up of the following modules, in order:

* env
* itensor
* module_rewriter
* op
* util

"""

from sys import modules as _modules
from types import ModuleType as _ModuleType
from importlib import import_module as _import_module


class _LazyModule(_ModuleType):

    def __init__(self, pkg, mod, asname=None):
        '''Lazy module 'pkg.mod' in package 'pkg'.'''
        self.__dct__ = {
            'loaded': False,
            'pkg': pkg,  # pkg
            'mod': mod,  # pkg.mod
            'asname': asname,  # alias
            }

    @classmethod
    def load(cls, pkg, mod, asname=None):
        if mod in _modules:
            return _modules[pkg]
        else:
            return cls(pkg, mod, asname)

    def __getattribute__(self, name):
        if name == '__dct__':
            return super(_LazyModule, self).__getattribute__(name)
        dct = self.__dct__
        mod = dct['mod']
        if dct['loaded']:
            m = _modules[mod]
        else:
            m = _import_module(mod)
            glbs = globals()
            pkg = dct['pkg']
            asname = dct['asname']
            if asname is None:
                glbs[pkg] = m = _modules[pkg]
            else:
                glbs[asname] = m
            dct['loaded'] = True
        return getattr(m, name)

#
# env
#
"""Implementation of Imperative  environment. All user-facing elements
of framework should go here.

Env: imperative environment.
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

numbers = _LazyModule.load('numbers', 'numbers')
np = _LazyModule.load('numpy', 'numpy', 'np')
# from .itensor import ITensor
# from .module_rewriter import ImperativeRewriter
# from .module_rewriter import ModuleRewriter
# from .op import Op
# from .util import get_current_device_string
# from .util import is_list_or_tuple
# from .util import shorten_device_string

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

#
# itensor
#
"""Implementation of ITensor for the imperative API."""

# amalgamated from __future__ import absolute_import
# amalgamated from __future__ import division
# amalgamated from __future__ import print_function
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import tensor_shape
# amalgamated from tensorflow.python.ops import array_ops
class ITensor(object):
  """This class is a Python wrapper over underlying persistent tensor object.
  It tries to provide some compatibility with existing Tensor objects while
  providing Python numeric interface that automatically run operations in
  associated TensorFlow runtime."""

  def __init__(self, env, handle):
    """Initialize ITensor.

    Args:
      env: imperative.Env object
      handle: session_ops.TensorHandle object
    """
    self.env = env
    self.handle = handle

  @property
  def tf_handle(self):
    """Give string handle representing this itensor in TF runtime.
    This string handle is suitable for feeding get_session_tensor op."""
    return self.handle.handle

  @property
  def dtype(self):
    """Tensorflow dtype of given tensor."""
    return self.handle._dtype

  @property
  def shape(self):
    """Dynamic shape of tensor."""
    shape_tensor = self.env.tf.shape(self)
    return tuple(shape_tensor.as_numpy())

  def as_numpy(self):
    """Convert current ITensor into numpy array."""

    return self.env.handle_to_numpy(self.handle)

  # Some user-generated ops call shape inference functions
  # For compatibility with those functions, make this Tensor act like an op
  # with 3 unknown-shaped inputs.
  @property
  def op(self):
    """Method for compatibility with Tensor."""
    node_def = graph_pb2.NodeDef()
    node_def.name = "imperative-dummy-node"
    node_def.input.extend(["dummy1", "dummy2", "dummy3"])

    dummy_input1 = array_ops.placeholder(self.dtype)
    dummy_input2 = array_ops.placeholder(self.dtype)
    dummy_input3 = array_ops.placeholder(self.dtype)
    dummy_op = tf_ops.Operation(node_def, tf_ops.Graph(), inputs=[dummy_input1,
                                                                  dummy_input2,
                                                                  dummy_input3])

    return dummy_op

  @property
  def name(self):
    return "nameless-itensor"

  @property
  def graph(self):
    return self.env.graph
  
  def eval(self):
    """Method for compatiblity with Tensor."""
    return self.as_numpy()

  # TODO(yaroslavvb): replace this with TensorShape(None) to avoid unexpected
  # run calls once all static shape inference functions support Unknown shape
  def get_shape(self):
    """Method for compatibility with Tensor."""
    shape_tensor = self.env.tf.shape(self)
    return tensor_shape.TensorShape(tuple(shape_tensor.as_numpy()))

  # Imperative tensors don't have static shape, but keep this method
  # for compatibility with array_ops.py
  def set_shape(self, _unused_shape):
    """Method for compatiblity with Tensor."""
    pass

  def __repr__(self):
    return "ITensor(%s, dtype=%s)" % (self.as_numpy(), self.dtype.name)

  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types

  def __add__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.add(self, other)

  def __radd__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.add(other, self)

  def __neg__(self):
    return self.env.tf.neg(self)


  def __sub__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.sub(self, other)

  def __rsub__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.sub(other, self)

  def __mul__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.mul(self, other)

  def __rmul__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.mul(other, self)

  def __div__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.div(self, other)

  def __rdiv__(self, other):
    if not isinstance(other, ITensor):
      other = self.env.numpy_to_itensor(other, dtype=self.dtype)
    return self.env.tf.div(other, self)

  # TODO(yaroslavvb): make div and truediv do different things?
  def __truediv__(self, other):
    return self.__div__(other)

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __bool__(self):
    return bool(self.as_numpy())

  def __nonzero__(self):
    return self.__bool__()

  def __lt__(self, other):
    return self.env.tf.less(self, other)

  def __le__(self, other):
    return self.env.tf.less_equal(self, other)

  def __eq__(self, other):
    return self.env.tf.equal(self, other)

  def __ne__(self, other):
    return self.env.tf.not_equal(self, other)

  def __gt__(self, other):
    return self.env.tf.greater(self, other)

  def __ge__(self, other):
    return self.env.tf.greater_equal(self, other)

  def __abs__(self):
    return self.env.tf.abs(self)
  
  def __getitem__(self, slice_spec):
    # TODO(yaroslavvb) re-use _SliceHelper from array_ops.py instead of copying
    # the code. Right now it is not possible because _SliceHelper is not
    # exported in tf namespace (_ prefixed functions are not included in
    # import *), hence it's not wrapped

    if not isinstance(slice_spec, (list, tuple)):
      slice_spec = [slice_spec]
    indices = []
    sizes = []
    squeeze_dims = []
    for dim, s in enumerate(slice_spec):
      if isinstance(s, slice):
        if s.step not in (None, 1):
          raise NotImplementedError(
              "Steps other than 1 are not currently supported")
        start = s.start if s.start is not None else 0
        if start < 0:
          raise NotImplementedError(
              "Negative start indices are not currently supported")
        indices.append(start)
        if s.stop is not None and s.stop < 0:
          raise NotImplementedError(
              "Negative stop indices are not currently supported")
        # NOTE(mrry): If the stop is not specified, Python substitutes
        #   sys.maxsize, which is typically (2 ** 63) - 1. Since Slice currently
        #   supports signed DT_INT32 arguments, we use -1 to specify that all
        #   elements should be captured.
        if s.stop is None or s.stop == sys.maxsize:
          sizes.append(-1)
        else:
          if start > s.stop:
            raise ValueError("Stop must be at least start")
          sizes.append(s.stop - start)
      elif s is Ellipsis:
        raise NotImplementedError("Ellipsis is not currently supported")
      else:
        try:
          s = int(s)
        except TypeError:
          raise TypeError("Bad slice index %s of type %s" % (s, type(s)))
        if s < 0:
          raise NotImplementedError("Negative indices are currently "
                                    "unsupported")
        indices.append(s)
        sizes.append(1)
        squeeze_dims.append(dim)
    sliced = self.env.tf.slice(self, indices, sizes)
    if squeeze_dims:
      return self.env.tf.squeeze(sliced, squeeze_dims=squeeze_dims)
    else:
      return sliced


#
# module_rewriter
#
"""Contains ModuleRewriter class + helper methods useful for writing custom
symbol_rewriter functions to be used with ModuleRewriter.
"""

# amalgamated from __future__ import absolute_import
# amalgamated from __future__ import division
# amalgamated from __future__ import print_function
ctypes = _LazyModule.load('ctypes', 'ctypes')
imp = _LazyModule.load('imp', 'imp')
inspect = _LazyModule.load('inspect', 'inspect')
six = _LazyModule.load('six', 'six')
sys = _LazyModule.load('sys', 'sys')
types = _LazyModule.load('types', 'types')
# from .op import ConstantOpWrapper
# from .op import ConstantValueWrapper
# from .op import ConvertToTensorWrapper
# from .op import OpDefLibraryWrapper
# from .util import get_symbol_file
# from .util import is_contextlib_wrapped_function
# from .util import make_cell

# x-version compatibility
# constant_op moved in a558c6e3b38846727873b5afbbc3ba309ae5dff5
try:
  from tensorflow.python.framework import op_def_library
except ImportError:
  from tensorflow.python.ops import op_def_library

class ModuleRewriter(object):
  """Object that controls rewriting of module. It allows to create a new
  version of a module where some symbols are replaced with new versions and
  all references to those symbols are updated."""

  def __init__(self, symbol_rewriter, module_prefix="newmodule."):
    """Initialize ModuleRewriter.

    Rewriting is done by taking a custom "symbol-rewriter" and applying it to
    all symbols referenced from module provided to later __call__, directly or
    indirectly. If symbol-rewriter returns non-None value, the entire module
    is copied, the affected symbol is replaced with output of symbol rewriter
    and all references to old module are updated to point to new module. If
    symbol-rewriter always returns None, module is not affected and original
    reference is returned. If module is copied, any modules referencing it
    will also be copied, and the references will be updated to point to new
    copy.

    Only function/module references are followed, which includes decorator-
    wrapped functions. References inside objects and types are not followed.
    This means that while object and type references are retained in new module,
    references they rely on are not updated and will continue pointing to the
    old module hierarchy.

    Args:
      symbol_rewriter: callable object that implements symbol rewriting. It
          should accepts a symbol (ie, a function) and return new symbol that
          acts as a replacement, or None to keep original symbol unchanged.
          The name of the symbol should remain unchanged because it's used
          to resolve references from other modules.
      module_prefix: a string that is prefixed to __name__ and __file__
          attributes of copied modules. Because we add new modules to
          sys.modules, this string must be non-empty.
    """

    assert module_prefix, "Module prefix must be non-empty"

    self.symbol_rewriter = symbol_rewriter
    self.module_prefix = module_prefix

    self._done_modules = {}  # dict of old_module->new_module
    self._module_stack = []  # stack of modules to detect cycles

    # initialize dictionary of modules and their filenames
    # this is needed for contextlib wrapped functions because they change
    # __module__ to __module__ of original function, so in order to update
    # references properly, we need to lookup their module using combination
    # of inspect.getsourcefile() and pre-computed dict
    self._module_fname_dict = {}
    for _unused_name, module in sys.modules.items():
      if hasattr(module, "__file__"):
        if module.__file__:
          fname = module.__file__
          if fname.endswith('.pyc'):  # strip .pyc into .py
            fname = fname[:-1]
          self._module_fname_dict[fname] = module

  def __call__(self, original_module):
    return self._rewrite_module(original_module)

  def _rewrite_module(self, original_module):
    """Apply symbol_rewriter to given module and its dependencies recursively
    and return the result. Copies of objects are made as necessary and original
    module remains unchanged.

    Args:
      original_module: module to rewrite.

    Returns:
      Copy of module hierarchy with rewritten symbols.
    """

    # system modules are missing __file__ attribute, and checking by
    # id is insufficient to prevent infinite loops, hence forbid missing
    # __file__
    if not hasattr(original_module, "__file__") or not original_module.__file__:
      self._done_modules[original_module] = original_module

    if original_module in self._done_modules:
      return self._done_modules[original_module]

    if original_module.__file__ in self._module_stack:
      return original_module

    self._module_stack.append(original_module.__file__)
    updated_symbols = {}  # symbols that got touched


    # Go over all symbols in a module to determine if module needs to be copied
    for symbol_name, symbol in original_module.__dict__.items():

      # Case 1: symbol is directly replaced by symbol_rewriter
      new_symbol = self.symbol_rewriter(symbol)
      if new_symbol:
        updated_symbols[symbol_name] = new_symbol

      # Case 2: symbol is a module which may be affected by symbol_rewriter
      elif isinstance(symbol, types.ModuleType):  # prevent infinite recursion
        if get_symbol_file(symbol) not in self._module_stack:
          new_symbol = self._rewrite_module(symbol)

          # copied modules always get a new name (prefixed with module_prefix)
          if new_symbol.__name__ != symbol.__name__:
            updated_symbols[symbol_name] = new_symbol

      # Case 3: contextlib-decorated functions have closures, treat them like
      # functions in new modules, with additional copying of closures
      elif is_contextlib_wrapped_function(symbol):
        assert len(symbol.__closure__) == 1
        wrapped_function = symbol.__closure__[0].cell_contents
        assert isinstance(wrapped_function, types.FunctionType), ("Only know "
            "how to rewrite simply wrapped functions")

        module_fname = inspect.getsourcefile(symbol)
        symbol_module = self._lookup_module_by_fname(module_fname)
        new_symbol_module = self._rewrite_module(symbol_module)
        closure_module = sys.modules[wrapped_function.__module__]
        assert closure_module
        new_closure_module = self._rewrite_module(closure_module)

        if new_closure_module.__name__ != closure_module.__name__:
          # don't know how to update both closure and symbol
          assert new_symbol_module.__name__ == symbol_module.__name__
          new_wrapped_function = (new_closure_module.__dict__
                                  [wrapped_function.__name__])
          new_symbol = copy_function_with_closure(symbol, new_symbol_module,
                                                  new_wrapped_function)
          updated_symbols[symbol_name] = new_symbol


      # Case 4: symbol is a function defined in a module which may be affected
      # by symbol rewriter
      elif hasattr(symbol, "__module__") and isinstance(symbol,
                                                        types.FunctionType):
        # If a function is defined in a different module, ie "import x from y"
        # copy "y" if necessary and update x to to point to new module
        if symbol.__module__ != original_module.__name__:
          symbol_file = get_symbol_file(symbol)
          if symbol_file and symbol_file not in self._module_stack:
            symbol_module = sys.modules[symbol.__module__]
            new_symbol_module = self._rewrite_module(symbol_module)

            if new_symbol_module.__name__ != symbol_module.__name__:
              updated_symbols[symbol_name] = new_symbol_module.__dict__[
                  symbol.__name__]

    # nothing was modified, so return module unchanged
    if not updated_symbols:
      self._done_modules[original_module] = original_module
      self._module_stack.pop()
      return original_module

    # module was modified, hence make a new copy
    new_module_name = self.module_prefix + original_module.__name__
    new_module = imp.new_module(new_module_name)
    new_module.__package__ = ""
    new_module.__file__ = self.module_prefix + original_module.__file__

    for symbol_name, symbol in original_module.__dict__.items():
      if symbol_name in ('__file__', '__name__', '__package__'):
        continue

      # it's a symbol we have created a new version of
      if symbol_name in updated_symbols:
        new_symbol = updated_symbols[symbol_name]

        if (hasattr(new_symbol, "__module__") and
            new_symbol.__module__ == original_module.__name__):
          new_symbol.__module__ = new_module.__name__

        new_module.__dict__[symbol_name] = new_symbol

      # it's a function whose definition wasn't updated
      elif isinstance(symbol, types.FunctionType):
        # if it's contextlib-wrapped function, its definition lies in different
        # module, since this definition wasn't updated, the different module
        # wasn't updated either, so retain original reference
        if is_contextlib_wrapped_function(symbol):
          new_symbol = symbol

        elif symbol.__module__ == original_module.__name__:
          new_symbol = copy_function(symbol, new_module)

        else:
          new_symbol = symbol
        new_module.__dict__[symbol_name] = new_symbol

      # it's a class
      elif (isinstance(symbol, six.class_types) and
            EXPERIMENTAL_SUPPORT_CLASSES):
        new_symbol = copy_class(symbol, new_module)
        new_module.__dict__[symbol_name] = new_symbol

      else:  # unsupported elements remain unchanged
        new_module.__dict__[symbol_name] = symbol

    sys.modules[new_module_name] = new_module
    self._done_modules[original_module] = new_module
    self._module_stack.pop()
    return new_module

  def _lookup_module_by_fname(self, fname):
    """Find module based on its __file__ attribute."""

    if fname.endswith('.pyc'):
      fname = fname[:-1]
    return self._module_fname_dict.get(fname, None)


class ImperativeRewriter(object):
  """A symbol rewriter that replaced all symbols relevant for imperative
  execution with corresponding imperative versions."""

  def __init__(self, env):
    self.env = env

  def __call__(self, symbol):
    # replace _op_lib_def in gen_.*_ops files
    if isinstance(symbol, op_def_library.OpDefLibrary):
      return OpDefLibraryWrapper(self.env, symbol)

    if isinstance(symbol, types.FunctionType):
      if (symbol.__name__ == 'convert_to_tensor' and
          symbol.__module__ == 'tensorflow.python.framework.ops'):
        return ConvertToTensorWrapper(self.env, symbol)

      if (symbol.__name__ == 'constant' and
          symbol.__module__ == 'tensorflow.python.framework.constant_op'):
        return ConstantOpWrapper(self.env, symbol)

      if (symbol.__name__ == 'constant_value' and
          symbol.__module__ == 'tensorflow.python.framework.tensor_util'):
        return ConstantValueWrapper(self.env, symbol)

def copy_function(old_func, updated_module):
  """Copies a function, updating it's globals to point to updated_module."""

  new_func = types.FunctionType(old_func.__code__, updated_module.__dict__,
                                name=old_func.__name__,
                                argdefs=old_func.__defaults__,
                                closure=old_func.__closure__)
  new_func.__dict__.update(old_func.__dict__)
  new_func.__module__ = updated_module.__name__
  return new_func

def copy_function_with_closure(old_func, updated_module,
                               updated_wrapped_func):
  """Copies a function, updating it's globals to point to updated_module.
  This works for singly-wrapped function (ie, closure has len 1).
  """

  assert old_func.__closure__ and len(old_func.__closure__) == 1

  cell = make_cell()
  PyCell_Set = ctypes.pythonapi.PyCell_Set

  # ctypes.pythonapi functions need to have argtypes and restype set manually
  PyCell_Set.argtypes = (ctypes.py_object, ctypes.py_object)

  # restype actually defaults to c_int here, but we might as well be explicit
  PyCell_Set.restype = ctypes.c_int

  PyCell_Set(cell, updated_wrapped_func)

  new_closure = (cell,)
  new_func = types.FunctionType(old_func.__code__, updated_module.__dict__,
                                name=old_func.__name__,
                                argdefs=old_func.__defaults__,
                                closure=new_closure)
  new_func.__dict__.update(old_func.__dict__)
  new_func.__module__ = updated_module.__name__
  return new_func


EXPERIMENTAL_SUPPORT_CLASSES = False
# Even when classes are being copied right now, objects like
# DefaultGraphStack refer to the old version of the class, and since
# get_default_graph is executed by DefaultGraphStack object, it will get the
# un-updated version of class, so for class copy support to work to support
# ops.colocate_with, must also enable object copying

def copy_class(old_class, updated_module):
  """Coplies a class, updating globals of all the functions and all
  functions wrapped with contextlib to point to updated_module."""

  new_dict = {}
  for name, entry in old_class.__dict__.items():
    if not isinstance(entry, types.FunctionType):
      new_dict[name] = entry
    else:
      if not entry.__closure__:
        new_dict[name] = copy_function(entry, updated_module)
      else:
        if len(entry.__closure__) != 1:
          new_dict[name] = entry
          continue

        assert len(entry.__closure__) == 1
        wrapped_function = entry.__closure__[0].cell_contents
        if not isinstance(wrapped_function, types.FunctionType):
          new_dict[name] = entry
          continue

        assert isinstance(wrapped_function, types.FunctionType)
        assert is_contextlib_wrapped_function(entry)

        new_wrapped_function = copy_function(wrapped_function, updated_module)
        new_closure = (make_cell(new_wrapped_function),)
        new_entry = types.FunctionType(entry.__code__, entry.__globals__,
                                       name=entry.__name__,
                                       argdefs=entry.__defaults__,
                                       closure=new_closure)
        new_dict[name] = new_entry

  return type(old_class.__name__, old_class.__bases__, new_dict)

#
# op
#
"""This module contains implementations of imperative replacement of various
TensorFlow functions. They mainly are used by module_rewriter while wrapping
tensorflow namespace. The central method of imperative wrapping is "apply_op"
method of OpDefLibraryWrapper, which provides a version of "apply_op" that
works with itensors instead of tensors.

Op: helper class that wraps env and piece of Graph into a callable Python object
OpDefLibraryWrapper: substitution for op_def_library in gen_.*_op files, it
    provides imperative-compatible version of apply_op
ConstantOpWrapper: replacement of constant_op.constant
ConvertToTensorWrapper: replacement of ops.convert_to_tensor
ConstantValueWrapper: replacement of tensor_util.constant_value
"""

# amalgamated from __future__ import absolute_import
# amalgamated from __future__ import division
# amalgamated from __future__ import print_function
# amalgamated from itensor import ITensor
# from .util import IsListParameter
# amalgamated from util import get_current_device_string
# amalgamated from util import is_list_or_tuple
# amalgamated from util import shorten_device_string
# amalgamated from util import shorten_device_string
# amalgamated from tensorflow.python.framework import dtypes
# amalgamated from tensorflow.python.framework import ops
# amalgamated from tensorflow.python.ops import session_ops
class Op(object):
  """Op represents an object which accepts itensors and returns itensors
  It turns incoming ITensors into TensorHandle objects, runs underlying op in
  env's session and wraps result in ITensor objects."""

  def __init__(self, env, input_holders, output_handle, name="op"):
    """Initialize Op.

    Args:
      env: imperative.Env object that is used to run this operation
      input_holders: dictionary of input_arg name to Placeholders or lists of
          Placeholders where corresponding input will be fed. Lists of
          holders are used for list input arguments like for Concat. This
          mapping is used to match keyword inputs in the __call__ method to
          their proper placeholders
      output_handle: a get_tensor_handle tensor or list of get_tensor_handle
          tensors that contain output of the op.
      name: human-readable name of op used for display
    """

    self.env = env
    self.input_holders = input_holders
    self.output_handle = output_handle
    self.name = name

  def __call__(self, **kwargs):
    """Feed ITensors into the op and return ITensor or list of ITensor result.
    """

    feed_dict = {}
    for argname in self.input_holders:
      itensor = kwargs[argname]
      holder = self.input_holders[argname]
      if is_list_or_tuple(holder):
        for subholder, subtensor in zip(holder, itensor):
          feed_dict[subholder] = subtensor.tf_handle
      else:
        feed_dict[holder] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    if is_list_or_tuple(tensor_handle):
      return [ITensor(self.env, t) for t in tensor_handle]
    else:
      return ITensor(self.env, tensor_handle)

  def __repr__(self):
    return "Op(%s)" % (str(self.name))


class OpDefLibraryWrapper(object):
  """Wrapper class that replaces OpDefLibrary instances in all gen.*ops
  modules. Used by module_rewriter."""

  def __init__(self, env, original_op_def_library):
    """Initialize OpDefLibraryWrapper.

    Args:
      env: imperative.Env object
      original_op_def_library: ops.OpDefLibrary object
    """
    self.env = env
    self.original_op_def_library = original_op_def_library

  def apply_op(self, op_type_name, name=None, **keywords):
    """Wrapper for op_def_library apply_op with caching.

    This method aims to be semantically identical to "apply_op" of OpDefLibrary
    but work with ITensor instead of Tensor objects.

    Brief overview

    1. Extract input arguments from keywords and convert Python types into
    corresponding itensors using type constraints of the corresponding OpDef
    2. Figure out OpDef that would've been constructed for this op if original
       op_def_library were called by looking at inferred/explicit attributes,
       argument device locations, and current device constext
    3. Fetch corresponding OpDef from cache if such OpDef was already
       constructed
    4. Otherwise construct OpDef and wrap it in Op object
    5. Save Op object in cache, and run it to produce itensor result
    """

    op_def = self._lookup_opdef_for_type(op_type_name)

    # names of input arguments, ie "x", "y" for Add op
    input_names = [arg.name for arg in op_def.input_arg]

    # convert any python inputs in keywords into ITensors
    convert_to_itensors_with_type_inference(op_def, keywords,
                                            self.env.numpy_to_itensor)

    current_device = get_current_device_string(self.env.g)
    key = create_opdef_key(op_def, keywords, current_device)
    op = self.env.cache_lookup(key)

    # Found op in cache, run it in return the results
    if op:
      return op(**keywords)

    # Couldn't find op in graph cache, create it and add to cache
    if self.env.PRINT_CACHE_MISSES:
      print("Imperative cache miss for %s" %(str(key)))


    # Graph construction overview:
    # The new operation must reproduce old operation, except that inputs
    # and outputs must be string tensor handles instead of Tensors
    # 1. Convert input string tensor handles into Tensors
    # 2. Run the op
    # 3. Convert output tensors into string tensor handles

    # prefix to use for node names in graph, like "Add.float32"
    if len(input_names) > 0 and isinstance(keywords[input_names[0]],
                                           ITensor):
      op_prefix = op_type_name + "."+keywords[input_names[0]].dtype.name
    else:
      op_prefix = op_type_name + ".no_dtype"

    # keywords for original apply_op, ITensor entries will be replaced with
    # Tensors
    opdeflib_keywords = dict(keywords)

    # Graph construction 1/3: inputs
    # replace ITensor inputs with tensorhandle->tensor converters
    with self.env.g.as_default():
      input_holders = {}  # placeholders for string tensor handle feeding

      for input_num, input_name in enumerate(sorted(input_names)):
        op_name = op_prefix + "." + str(input_num)
        itensor_input = keywords[input_name]
        # single tensor input
        if isinstance(itensor_input, ITensor):
          holder, tensor = session_ops.get_session_tensor(
              itensor_input.tf_handle, itensor_input.dtype, name=op_name)
          input_holders[input_name] = holder
          opdeflib_keywords[input_name] = tensor

        # list input, such as for tf.concat, add converter for each element
        else:
          assert is_list_or_tuple(itensor_input)
          holder_list = []
          tensor_list = []
          for subinput_num, subinput in enumerate(itensor_input):
            op_name = op_name + "_" + str(subinput_num)
            holder, tensor = session_ops.get_session_tensor(subinput.tf_handle,
                                                            subinput.dtype,
                                                            name=op_name)
            holder_list.append(holder)
            tensor_list.append(tensor)
            opdeflib_keywords[input_name] = tensor_list
          input_holders[input_name] = holder_list

      # Graph construction 2/3: op
      # call original apply_op to create the op
      output = self.original_op_def_library.apply_op(op_type_name,
                                                     name=op_prefix+".op",
                                                     **opdeflib_keywords)


      # Graph construction 3: outputs
      # attach tensor->tensorhandle conversion to outputs
      op_name = op_prefix+".out"

      # single Tensor output
      if isinstance(output, ops_lib.Tensor):
        output_handle = session_ops.get_session_handle(output,
                                                       op_name+".handle")
      # operation output like with.control_dependencies
      elif isinstance(output, ops_lib.Operation):
        assert False, "Imperative mode only supports ops that produce tensors"

      else:  # list of Tensors, such as for tf.split
        assert is_list_or_tuple(output)
        output_handle = []
        for output_num, output_tensor in enumerate(output):
          op_name = op_name + "_" + str(output_num)
          output_single_handle = session_ops.get_session_handle(output_tensor,
                                                                (op_name+
                                                                 ".handle"))
          output_handle.append(output_single_handle)

    # save our newly created op in cache
    op = Op(self.env, input_holders, output_handle)
    self.env.cache_add(key, op)

    # execute the op
    return op(**keywords)

  def _lookup_opdef_for_type(self, op_type_name):
    """Retrieves OpDef proto for given op type."""

    return self.original_op_def_library._ops[op_type_name].op_def


class ConstantOpWrapper(object):
  """A callable object that mirrors tf.constant."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, *args, **kwargs):
    return self.env.constant(*args, **kwargs)


class ConvertToTensorWrapper(object):
  """A callable object that mirrors tf.convert_to_tensor in Imperative
  environment."""

#  def __init__(self, namespace, env, symbol_name):
  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, value, dtype=None, name=None, as_ref=False):
    if isinstance(value, ITensor):
      return value
    return self.env.numpy_to_itensor(value, dtype)


class ConstantValueWrapper(object):
  """A callable object that mirrors tensor_util.constant_value in Imperative
  environment."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, itensor):
    return itensor.as_numpy()


def create_opdef_key(op_def, keywords, op_device):
  """Construct unique key representing current opdef. Key is constructed from
  devices of input itensors, location of op and implicit/explicit attributes of
  the OpDef."""

  # extract inferred attributes
  input_names = [input_arg.name for input_arg in op_def.input_arg]
  attr_names = [attr.name for attr in op_def.attr]

  # extract inferred attributes from input types and sizes
  attrs = {}
  for input_arg in op_def.input_arg:
    input_itensor = keywords[input_arg.name]
    if input_arg.type_attr:
      if input_arg.type_attr in attrs:
        assert attrs[input_arg.type_attr] == input_itensor.dtype
      else:
        # for list parameter, take dtype of first entry in list
        if IsListParameter(input_arg):
          assert len(input_itensor) > 0
          attrs[input_arg.type_attr] = input_itensor[0].dtype
        else:
          attrs[input_arg.type_attr] = input_itensor.dtype
    if input_arg.number_attr:
      attrs[input_arg.number_attr] = len(input_itensor)
    if input_arg.type_list_attr:
      attrs[input_arg.type_list_attr] = tuple(i.dtype for i in
                                              input_itensor)
  # extract explicit attributes
  for key in keywords:
    if key in input_names:
      continue
    assert key not in attrs, ("Can't specify an inferred attr " +
                              key)
    attrs[key] = keywords[key]


  def extract_device(itensor):
    """Extract device like "gpu:0" from itensor."""
    device_name = session_ops.TensorHandle._get_device_name(itensor.tf_handle)
    return shorten_device_string(device_name)

  # extract input devices
  input_devices = {}
  for name in input_names:
    itensor = keywords[name]
    if isinstance(itensor, list) or isinstance(itensor, tuple):
      device = tuple(extract_device(subtensor) for subtensor in itensor)
    else:
      device = extract_device(itensor)
    input_devices[name] = device

  assert set(attr_names) == set(attrs.keys())
  key = [op_def.name]

  for attr in sorted(attrs.keys()):
    key.append(str(attr))
    if isinstance(attrs[attr], dtypes.DType):
      attr_name = str(attrs[attr].name)
    else:
      attr_name = str(attrs[attr])
    key.append(attr_name)

  for name in sorted(input_names):
    key.append(str(name))
    key.append(str(input_devices[name]))

  key.append(str(op_device))
  hashable_key = tuple(key)

  assert hash(hashable_key)
  return hashable_key

def is_itensor_or_itensors(value):
  """Returns true if argument is imperative Tensor or list/tuple of Tensors."""

  if isinstance(value, ITensor):
    return True
  elif isinstance(value, list) or isinstance(value, tuple):
    for potential_tensor in value:
      if not isinstance(potential_tensor, ITensor):
        return False
    return True
  else:
    return False

def convert_to_itensors_with_type_inference(op_def, keywords,
                                            numpy_to_itensor):
  """When elements of entries are provided as Python types, convert them to
  itensors while following type constraints in op_def."""

  arg_names = [arg.name for arg in op_def.input_arg]
  if all(is_itensor_or_itensors(keywords[n]) for n in arg_names):
    return

  attrs = {}

  # Stage 1, go over input arguments, and initialize type attributes from
  # ITensor dtypes

  # dictionary like "values" -> input_arg {name: "values", type_attr: "T"}
  input_args = {arg.name: arg for arg in op_def.input_arg}
  for name in arg_names:
    itensor = keywords[name]
    if IsListParameter(input_args[name]):
      assert isinstance(itensor, list) or isinstance(itensor, tuple)
      type_attr_name = input_args[name].type_attr
      if type_attr_name:
        for subtensor in itensor:
          if isinstance(subtensor, ITensor):
            if type_attr_name in attrs:
              assert attrs[type_attr_name] == subtensor.dtype
            else:
              attrs[type_attr_name] = subtensor.dtype
    else:
      if isinstance(itensor, ITensor):
        type_attr_name = input_args[name].type_attr
        if type_attr_name:
          if type_attr_name in attrs:
            assert attrs[type_attr_name] == itensor.dtype
          else:
            attrs[type_attr_name] = itensor.dtype

  # Stage 2, go over input arguments again, and convert Python types
  # to inferred type attributes. If no type attribute was inferred
  # (such as the case when all inputs are Python), use default conversion
  # and hope they are correct types (don't enforce type consistency)
  for name in arg_names:
    itensor = keywords[name]
    type_attr_name = input_args[name].type_attr
    inferred_dtype = attrs.get(type_attr_name, None)

    if IsListParameter(input_args[name]):
      for i, subtensor in enumerate(itensor):
        if not isinstance(subtensor, ITensor):
          itensor[i] = numpy_to_itensor(itensor[i], inferred_dtype)
    else:
      if not isinstance(itensor, ITensor):
        keywords[name] = numpy_to_itensor(itensor, inferred_dtype)


#
# util
#
"""Utilities used by internal implementation of imperative mode."""

# amalgamated from __future__ import absolute_import
# amalgamated from __future__ import division
# amalgamated from __future__ import print_function
# amalgamated inspect
# amalgamated sys
# amalgamated types
class _DeviceCaptureOp(object):
  def __init__(self):
    self.device = None

  def _set_device(self, device):
    self.device = device

def get_current_device_string(graph):
  """Returns short device string like "cpu:0 used by current graph."""
  op = _DeviceCaptureOp()
  graph._apply_device_functions(op)
  if op.device:
    long_device_string = op.device.to_string()
    return shorten_device_string(long_device_string)
  else:
    return "None"

def shorten_device_string(long_device_string):
  """Turns long device string into short string like "gpu:0" . """
  start_pos = long_device_string.index("/device:")
  assert start_pos >= 0
  short_device_string = long_device_string[start_pos+len("/device:"):]
  assert short_device_string
  return short_device_string.lower()


def flatten_list(l):
  """Removes one layer of nesting from the list."""

  new_list = []
  for element in l:
    if isinstance(element, list) or isinstance(element, tuple):
      new_list.extend(element)
    else:
      new_list.append(element)
  return new_list


def IsListParameter(arg):
  """Returns if ArgDef represents a list parameter."""
  if arg.number_attr:
    return True
  elif arg.type_list_attr:
    return True
  return False

def is_list_or_tuple(value):
  return isinstance(value, list) or isinstance(value, tuple)


def is_contextlib_wrapped_function(symbol):
  """Check if this is a contextlib-wrapped function."""
  if not isinstance(symbol, types.FunctionType):
    return False
  try:  # try catch because getsourcefile fails with various errors
    fname = inspect.getsourcefile(symbol)
    if (not fname.endswith('contextlib.py') and
        not fname.endswith('contextlib.pyc')):
      return False
    if not symbol.__closure__:
      return False
    return True
  except:
    return False


def make_cell(val=None):
  """Helper function to make closure cell since there's no constructor."""
  x = val
  def closure():
    return x
  return closure.__closure__[0]

def get_symbol_file(symbol):
  """Returns filename of symbol definition, empty string if not available."""

  if hasattr(symbol, "__file__"):
    return symbol.__file__
  elif not isinstance(symbol, types.ModuleType):
    try:
      symbol_module = sys.modules[symbol.__module__]
      return symbol_module.__file__
    except (AttributeError, KeyError):
      return ""


def get_symbol_name(symbol):
  """Returns __name__ attribute or empty string if not available."""
  if hasattr(symbol, "__name__"):
    return symbol.__name__
  else:
    return ""

def print_gdef_diff(gdef1, gdef2):
  """Prints nodes in gdef2 that aren't in gdef1."""
  
  print("GraphDef difference")
  print("-"*80)
  dict1 = {node.name: node for node in gdef1.node}
  dict2 = {node.name: node for node in gdef2.node}
  names1 = set(dict1.keys())
  names2 = set(dict2.keys())
  if names1 == names2:
    return
  for name in sorted(names2.difference(names1)):
    print(dict2[name])

def self_test():
  import tensorflow as tf
  env = Env(tf)
  assert env.tf.add(1, 2) == 3
  print("Self test passed.")
