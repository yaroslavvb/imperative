"""Script to do some post-processing changes to result of amalgamate.py, see
make_package.sh for example of usage."""

import os, sys, re

def comment_out_the_line(line):
  return "# "+line

if __name__ == "__main__":
  outfile = open(sys.argv[2], "w")
  for line in open(sys.argv[1]):
    # lines like
    # from __future__ import absolute_import
    if line.startswith("from __future__ import"):
      outfile.write(comment_out_the_line(line))
      # lines like
      # from .itensor import ITensor
    elif re.match("^from\s+[.].*", line):
      outfile.write(comment_out_the_line(line))
    else:
      outfile.write(line)

  # add a simple self-test to the end
  outfile.write("""def self_test():
  import tensorflow as tf
  env = Env(tf)
  assert env.tf.add(1, 2) == 3
  print("Self test passed.")
""")

  outfile.close()

  
