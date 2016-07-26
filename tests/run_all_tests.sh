#!/bin/sh
BASEDIR=$(dirname "$0")
echo "Running simple test"
python $BASEDIR/simple.py
echo "Running env_test.py"
python $BASEDIR/env_test.py
echo "Running env_cache_test.py"
python $BASEDIR/env_cache_test.py
echo "Running extra/batchnorm_test.py"
python $BASEDIR/extra/batchnorm_test.py
echo "Running extra/histogram_ops_test.py"
python $BASEDIR/extra/histogram_ops_test.py
echo "Running extra/math_ops_test.py"
python $BASEDIR/extra/math_ops_test.py
echo "Running extra/nn_test.py"
python $BASEDIR/extra/nn_test.py
echo "Running itensor_test.py"
python $BASEDIR/itensor_test.py
echo "Running module_rewriter_test.py"
python $BASEDIR/module_rewriter_test.py
echo "Running lbfgs_test.py"
python $BASEDIR/lbfgs_test.py
echo "Running mnist_inference_test.py"
python $BASEDIR/mnist_inference_test.py
# echo "Running image_ops_double_test"
# python extra/image_ops_double_test.py
# echo "Running image_ops_test"
# python extra/image_ops_test.py
