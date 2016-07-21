#!/bin/sh
BASEDIR=$(dirname "$0")
echo "Running env_test.py"
python $BASEDIR/env_test.py
echo "Running env_cache_test.py"
python $BASEDIR/env_cache_test.py
echo "Running extra_tests/batchnorm_test.py"
python $BASEDIR/extra_tests/batchnorm_test.py
echo "Running extra_tests/histogram_ops_test.py"
python $BASEDIR/extra_tests/histogram_ops_test.py
echo "Running extra_tests/math_ops_test.py"
python $BASEDIR/extra_tests/math_ops_test.py
echo "Running extra_tests/nn_test.py"
python $BASEDIR/extra_tests/nn_test.py
echo "Running itensor_test.py"
python $BASEDIR/itensor_test.py
echo "Running module_rewriter_test.py"
python $BASEDIR/module_rewriter_test.py
echo "Running lbfgs_test.py"
python $BASEDIR/lbfgs_test.py
echo "Running mnist_inference_test.py"
python $BASEDIR/mnist_inference_test.py
# echo "Running image_ops_double_test"
# python extra_tests/image_ops_double_test.py
# echo "Running image_ops_test"
# python extra_tests/image_ops_test.py
