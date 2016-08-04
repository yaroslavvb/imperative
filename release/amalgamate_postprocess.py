"""Script to do some post-processing changes to result of amalgamate.py, see
make_package.sh for example of usage."""

import os, sys, re

if __name__ == "__main__":
  outfile = open(sys.argv[2], "w")
  for line in open(sys.argv[1]):
    outfile.write(line)

  # add a simple self-test to the end
  outfile.write("""def self_test():
  import tensorflow as tf
  env = Env(tf)
  assert env.tf.add(1, 2) == 3
  print("Self test passed.")
""")

  outfile.close()

  
