#!/usr/bin/env python
import time

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

try:
    set_random_seed = tf.set_random_seed
except AttributeError:
    set_random_seed = tf.random.set_seed
set_random_seed(42)

try:
    random_normal = tf.random_normal
except AttributeError:
    random_normal = tf.random.normal

try:
    Session = tf.Session
except AttributeError:
    Session = tf.compat.v1.Session

A = random_normal([10000,10000])
B = random_normal([10000,10000])
start_time = time.time()
with Session() as sess:
    print(sess.run(tf.reduce_sum(tf.matmul(A,B))))
print(" took {} seconds ".format(time.time() - start_time))
