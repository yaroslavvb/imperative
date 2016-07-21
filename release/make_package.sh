#!/bin/sh
CURDIR=`pwd`
BASEDIR=$(dirname "$0")
cd $BASEDIR/../..
echo `pwd`
# amalgamate all files into single file
python3 imperative/release/amalgamate.py imperative

# some fixes (remove __future__, errant import statements)
python imperative/release/amalgamate_postprocess.py imperative/__amalgam__.py imperative/imperative.py

# amalgamate script creates __init__, get rid of it
rm imperative/__init__.py
echo 'from env import *' > imperative/__init__.py

cd $CURDIR
