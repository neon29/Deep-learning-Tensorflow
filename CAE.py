# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:59:01 2017

@author: Florian

Convolutional autoencoder on MNIST dataset
"""

# to be included in all python tensorflow files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from activations import lrelu



# import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)



def conv_2D(x, w, b, stride=1, padding='SAME'):
    '''
    2D convolution
    x: tensor of shape (batch, height, width, channel) -> 
    w: tensor of shape (f_width, f_height, channels_in, channels_out) -> weights
    b: tensor of shape (channels_out) -> biases
    '''
    # convolution
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    # add biases
    x = tf.nn.bias_add(x, b)
    
    return x
    


def CAE(x):
    '''
    As there is no unpooling layer in tensorflow, we use a stride of 2 in both
    pooling/unpooling layer, to decrease feature maps size instead of pooling
    '''
    with tf.variable_scope('CAE', reuse=True):
        
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        print(x.get_shape().as_list())

        
        # first convolutional layer + pooling
        conv_layer_1 = tf.nn.sigmoid(conv_2D(x, tf.get_variable('conv1'), 
                                     tf.get_variable('conv1_bias'), stride=2))
        
        print(conv_layer_1.get_shape().as_list())
        
        # second one
        conv_layer_2 = tf.nn.sigmoid(conv_2D(conv_layer_1, tf.get_variable('conv2'), 
                                     tf.get_variable('conv2_bias'), stride=2))
        
        print(conv_layer_2.get_shape().as_list())
           
        
        # first "deconvolutional" layer
        deconv_layer_1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(conv_layer_2, 
                                         tf.get_variable('conv2'), 
                                        output_shape=tf.pack([tf.shape(x)[0], 14, 14, 10]),
                                        strides=[1, 2, 2, 1],
                                        padding='SAME'), tf.get_variable('deconv1_bias')))
                                                
        print(deconv_layer_1.get_shape().as_list())   
                                    
        # second one                                        
        output = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(deconv_layer_1, 
            tf.get_variable('conv1'), 
            output_shape=tf.pack([tf.shape(x)[0], 28, 28, 1]),
            strides=[1, 2, 2, 1],
            padding='SAME'), tf.get_variable('out_bias')))
        

        print(output.get_shape().as_list())   
                                     
        return output
                         
 

nb_samples_in_batch = 128
nb_epoch = 2
input_shape = 784
learning_rate = 0.01
                        
session = tf.InteractiveSession()     
                         
with tf.variable_scope('CAE', reuse=False):
    _ = tf.get_variable('conv1', initializer=tf.random_normal([3, 3, 1, 10]))
    #_ = tf.get_variable('conv1', initializer=tf.random_uniform([3, 3, 1, 10], -1.0/np.sqrt(1), 1.0/np.sqrt(1)))
                
    _ = tf.get_variable('conv2', initializer=tf.random_normal([3, 3, 10, 10]))
    #_ = tf.get_variable('conv2', initializer=tf.random_uniform([3, 3, 10, 10], -1.0/np.sqrt(10), 1.0/np.sqrt(10)))

    _ = tf.get_variable('conv1_bias', initializer=tf.random_normal([10]))
    _ = tf.get_variable('conv2_bias', initializer=tf.random_normal([10]))
    
    _ = tf.get_variable('deconv1_bias', initializer=tf.random_normal([10]))
    _ = tf.get_variable('out_bias', initializer=tf.random_normal([1]))
    
                
   
X = tf.placeholder(tf.float32, [None, input_shape])
Y_true = tf.reshape(X, shape=[-1, 28, 28, 1])
Y_pred = CAE(X)


# cost function
cost = tf.reduce_sum(tf.square(Y_pred-Y_true))

# optimizer
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

session.run(init)

nb_mini_batches = mnist.train.num_examples // nb_samples_in_batch

# loop over epoches
for i_epoch in range(nb_epoch):
    # loop over mini-batches
    for i_batch in range(nb_mini_batches):
        batch_x, _ = mnist.train.next_batch(nb_samples_in_batch)
        
        session.run(optim, feed_dict={X: batch_x})
        
        if i_batch % 10 == 0:
            c = session.run(cost, feed_dict={X: batch_x})
            print('epoch %i - batch %i - cost=%f' % (i_epoch, i_batch, c))
            
            
# test reconstruction
test_x, _ = mnist.test.next_batch(5)
res = session.run(Y_pred, feed_dict={X: test_x })

fig, axes = plt.subplots(2, res.shape[0])

for i in range(res.shape[0]):
    axes[0][i].imshow(res[i, ...].reshape((28, 28)))
    axes[1][i].imshow(test_x[i, :].reshape((28, 28)))

        
        
        
        
    
    
    
    
    
    
    

        