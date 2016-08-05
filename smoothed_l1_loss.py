#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def smoothed_l1_loss(input_tensor):
    absval = tf.abs(input_tensor)
    ind = tf.to_int32(absval > 1)
    inner, outer = tf.dynamic_partition(absval, ind, 2)
    loss = tf.reduce_sum(0.5 * tf.square(inner)) + \
        tf.reduce_sum(outer - 0.5)
    return loss

if __name__ == "__main__":
    import numpy as np
    data = np.random.randn(1, 3, 2, 2) * 2
    input = tf.Variable(data)
    loss = smoothed_l1_loss(input)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    val1 = sess.run(loss)

    absval = abs(data)
    outer = absval > 1
    inner = absval <= 1
    val2 = np.sum(0.5 * np.square(absval[inner])) + np.sum(absval[outer] - 0.5)

    assert abs(val1 - val2) < 1e-5, "Can not validate smoothed_l1_loss"
