
# coding: utf-8

# In[6]:

import tensorflow as tf
import numpy as np


# In[2]:

from tensorflow.models.image.cifar10 import cifar10


# In[9]:

# inp: N x H x W x C
# filter: H x W x C x N
def ReLU():
    return tf.nn.relu

def Identity():
    return tf.identity

def Sigmoid():
    return tf.sigmoid

def Tanh():
    return tf.tanh

def Softmax():
    return tf.nn.softmax


# In[ ]:

def XavierNormal(gain=0.01):
    def sample(fan_in, fan_out, receptive_field_size):
        stddev = gain * tf.sqrt(2 / ((fan_in + fan_out)) * receptive_field_size)
        return tf.random_normal_initializer(mean=0.0, stddev=stddev)
    return sample


# In[5]:

def ConvLayer(inp, name, num_units, kernel_size, stride, pad='SAME', 
              Winit=XavierNormal(), binit=None, nonlinearity=Identity):
    shape = inp.get_shape().as_list()
    in_channel = shape[3]
    
    Wshape = [kernel_size, kernel_size, in_channel, num_units]
    fan_in = in_channel
    fan_out = num_units
    rfs = kernel_size * kernel_size
        
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', Wshape, Winit(fan_in, fan_out, rfs))
        conv = tf.nn.conv2d(inp, W, strides=[1, stride, stride, 1], pad=pad)
        if binit != None:
            b = tf.get_variable('biases', [num_units], binit)
            conv = tf.nn.bias_add(conv, b)
        
        return nonlinearity(conv)


# In[7]:

def FcLayer(inp, name, num_units, Winit=XavierNormal(), binit=None, nonlinearity=Identity):
    shape = inp.get_shape().as_list()
    feature_size = shape[1]
    
    if len(shape) > 2:
        inp = FlattenLayer(inp)
        feature_size = np.prod(shape[1:])
    
    fan_in =
    fan_out = 
    rfs =
    
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', [feature_size, num_units], Winit(fan_in, fan_out, rfs))
        fc = tf.matmul(inp, W)
        if binit != None:
            b = tf.get_variable('biases', [num_units], binit)
            fc = tf.nn.bias_add(fc, b)
        
        return nonlinearity(fc)


# In[ ]:



