#!/bin/sh
CURDIR=`pwd`
BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo `pwd`

# amalgamate script creates __init__, get rid of it
echo "# amalgamate exclude power_benchmark" > imperative/__init__.py
echo "from env import *" >> imperative/__init__.py

# amalgamate all files into single file
python3 release/amalgamate.py imperative
# python3 ~/amalgamate.git/amalgamate/amalgamate.py

# some fixes (remove __future__, errant import statements)
python release/amalgamate_postprocess.py imperative/__amalgam__.py release/imperative.py

# add missing import
echo "from env import *" >> imperative/__init__.py

cd $CURDIR
