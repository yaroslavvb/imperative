"""Contains ModuleRewriter class + helper methods useful for writing custom
symbol_rewriter functions to be used with ModuleRewriter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import imp
import inspect
import six
import sys
import types

from .op import ConstantOpWrapper
from .op import ConstantValueWrapper
from .op import ConvertToTensorWrapper
from .op import OpDefLibraryWrapper
from .util import get_symbol_file
from .util import is_contextlib_wrapped_function
from .util import make_cell

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
