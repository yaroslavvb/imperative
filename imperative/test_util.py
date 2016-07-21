"""Test utils for tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from tensorflow.core.protobuf import config_pb2

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.client import device_lib

try:
  import tensorflow.contrib.imperative as imperative
except:
  import imperative
  
class ImperativeTestCase(tf_test_util.TensorFlowTestCase):
  """Base class for tests that need to test TensorFlow in imperative mode.
  """

  def __init__(self, methodName="runTest"):
    super(ImperativeTestCase, self).__init__(methodName)

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def haveGpu0(self):
    device_names = [d.name for d in device_lib.list_local_devices()]
    return "/gpu:0" in device_names

  @contextlib.contextmanager
  def test_env(self, tf=None):
    """Returns a imperative Environment for use in executing tests.

    Example:

      class MyOperatorTest(test_util.TensorFlowTestCase):
        def testMyOperator(self):
          with self.test_env(tf):
            valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = MyOperator(valid_input)
            self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            invalid_input = [-1.0, 2.0, 7.0]
            with self.assertRaisesOpError("negative input not supported"):
              MyOperator(invalid_input)

    Returns:
      An Env object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """

    def prepare_config():
      config = config_pb2.ConfigProto()
      config.graph_options.optimizer_options.opt_level = -1
      return config

    env = imperative.Env.get_global_default_env()
    if env is None:
      env = imperative.Env(tf, config=prepare_config())

    with env.g.as_default():
      yield env


class TensorFlowTestCase(ImperativeTestCase):
  """Class for porting tests that expect TensorFlowTestCase."""

  @contextlib.contextmanager
  def test_session(self, use_gpu=False):
    """Compatibility method for test_util."""

    env = imperative.Env.get_global_default_env()
    assert env, "Must initialize Env before using test_session"

    with env.g.as_default():
      imperative_session = ImperativeSession()
      yield imperative_session


class ImperativeSession(object):
  """Imperative-mode replacement of Session."""

  def run(self, fetches, *_unused_args, **_unused_kwargs):
    return [itensor.as_numpy() for itensor in fetches]
