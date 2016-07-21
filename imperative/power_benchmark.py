import tensorflow as tf
import time

try:
  from tensorflow.contrib import imperative
except:
  import imperative

env = imperative.Env(tf)

n = 2024
num_iters = 10
print("Trying %d power iterations with %d x %d matrices" %(num_iters, n, n))

n_ops = n**3+n*n*(n-1)  # n**3 muls, n**2*(n-1) adds
a = env.tf.ones((n, n), dtype=tf.float32)/n
b = env.tf.ones((n, n), dtype=tf.float32)/n
speeds = []
for i in range(num_iters):
    start_time = time.time()
    b = env.tf.matmul(a, b)
    elapsed_time = time.time() - start_time
    speed = n_ops/elapsed_time/10**9
    speeds.append(speed)
    print("%s T ops/sec" %(speeds[-1]))
    
print("max speed: %s T ops/sec" %(max(speeds)))
expected_sum = n
actual_sum = env.tf.reduce_sum(b)
print("relative error: %s" % (abs(expected_sum-actual_sum)/expected_sum))

