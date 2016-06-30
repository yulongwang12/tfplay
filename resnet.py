
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

from tensorflow.models.image.cifar10 import cifar10


# In[3]:

from tensorflow.contrib.layers import avg_pool2d, max_pool2d, batch_norm, conv2d, fully_connected


# In[4]:

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


# In[5]:

def bottleneck(inp, filters, downpool, upchannel):
    if downpool:
        res_a = conv2d(inp, filters[0], stride=2, kernel_size=1,                        normalizer_fn=batch_norm,                        activation_fn=tf.nn.relu)
    else:
        res_a = conv2d(inp, filters[0], stride=1, kernel_size=1,                        normalizer_fn=batch_norm,                        activation_fn=tf.nn.relu)

    res_b = conv2d(res_a, filters[1], stride=1, kernel_size=3,                    normalizer_fn=batch_norm,                    activation_fn=tf.nn.relu)

    res_c = conv2d(res_b, filters[2], stride=1, kernel_size=1,                    normalizer_fn=batch_norm,                    activation_fn=None)

    if upchannel:
        if downpool:
            inp = conv2d(inp, filters[2], stride=2, kernel_size=1,                          normalizer_fn=batch_norm,                          activation_fn=None)
        else:
            inp = conv2d(inp, filters[2], stride=1, kernel_size=1,                          normalizer_fn=batch_norm,                          activation_fn=None)

    out = tf.nn.relu(inp + res_c)

    return out


# In[12]:

def resblock(inp, filters, downpool, upchannel):
    if downpool:
        res_a = conv2d(inp, filters, stride=2, kernel_size=3,                        normalizer_fn=batch_norm,                        activation_fn=tf.nn.relu)
    else:
        res_a = conv2d(inp, filters, stride=1, kernel_size=3,                        normalizer_fn=batch_norm,                        activation_fn=tf.nn.relu)

    res_b = conv2d(res_a, filters, stride=1, kernel_size=3,                    normalizer_fn=batch_norm,                    activation_fn=None)

    if upchannel:
        if downpool:
            inp = conv2d(inp, filters, stride=2, kernel_size=3,                          normalizer_fn=batch_norm,                          activation_fn=None)
        else:
            inp = conv2d(inp, filters, stride=1, kernel_size=3,                          normalizer_fn=batch_norm,                          activation_fn=None)

    out = tf.nn.relu(inp + res_b)
    return out


# In[21]:

def cifar10_resnet(inp, n=3):
    net = conv2d(inp, num_outputs=16, kernel_size=3, stride=1,                    normalizer_fn=batch_norm,                    activation_fn=tf.nn.relu)

    # 32x32, 16 units
    for i in range(n):
        net = resblock(net, 16, downpool=False, upchannel=False)

    # 16x16, 32 units
    net = resblock(net, 32, downpool=True, upchannel=True)
    for i in range(n-1):
        net = resblock(net, 32, downpool=False, upchannel=False)

    # 8x8, 64 units
    net = resblock(net, 64, downpool=True, upchannel=True)
    
    for i in range(n-1):
        net = resblock(net, 64, downpool=False, upchannel=False)

    net = avg_pool2d(net, kernel_size=6, stride=1)
    
    net = fully_connected(net, num_outputs=10,                           activation_fn=None)
    
    out = tf.nn.softmax(tf.reshape(net, [FLAGS.batch_size, -1]))
    
    return out


# In[22]:

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        images, labels = cifar10.distorted_inputs()
        
        logits = cifar10_resnet(images)
        
        loss = cifar10.loss(logits, labels)
        
        train_op = cifar10.train(loss, global_step)
        
        summary_op = tf.merge_all_summaries()
        
        init = tf.initialize_all_variables()
        
        sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        tf.train.start_queue_runners(sess=sess)
        
        for step in xrange(FLAGS.max_steps):
            _, loss_value = sess.run([train_op, loss])
            
            if step % 10 == 0:
                print 'step %d, loss = %.3f' % (step, loss_value)


# In[28]:

if __name__ == '__main__':
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


# In[ ]:



