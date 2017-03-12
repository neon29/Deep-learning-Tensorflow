# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:17:22 2017

@author: Florian

Implement convolutional neural networks

"""

# to be included in all python tensorflow files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def conv_2D(x, w, b, stride=1):
    '''
    2D convolution
    x: tensor of shape (batch, height, width, channel) -> 
    w: tensor of shape (f_width, f_height, channels_in, channels_out) -> weights
    b: tensor of shape (channels_out) -> biases
    '''
    # convolution
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    # add biases
    x = tf.nn.bias_add(x, b)
    
    return x
    
def max_pooling_2D(x, stride=2):
    '''
    max-pooling
    x: output of conv_2D
    '''
    
    x = tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    return x
    

def CNN(x, dropout=None, batch_normalization=None):
    '''
    Build a CNN
    x: tensor of shape (batch, n_inputs) = (batch, 784) for MNIST
    weights: dict of weights
    biases: dict of biases
    '''
    with tf.variable_scope('test', reuse=True):
        # conv_2D accepts shape (batch, height, width, channel) as input so
        # reshape it
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        
        # convolution 1
        conv_1 = conv_2D(x, tf.get_variable('conv1'), tf.get_variable('conv1_bias'))
        
        # max pooling
        conv_1 = max_pooling_2D(conv_1)
        
        # convolution 2
        conv_2 = conv_2D(conv_1, tf.get_variable('conv2'), tf.get_variable('conv2_bias'))
        
        # max pooling 
        conv_2 = max_pooling_2D(conv_2)
        
        # flatten tensor to be used in fully connected layer
        conv_2_flat = tf.reshape(conv_2, shape=[-1, tf.get_variable('fc1').get_shape().as_list()[0]])
        
        # fully connected layer 1
        fc_1 = tf.add(tf.matmul(conv_2_flat, tf.get_variable('fc1')), tf.get_variable('fc1_bias'))
        
        # relu (recently considered to be better than sigmoid)
        fc_1 = tf.nn.relu(fc_1)
        
        # apply dropout if required`
        if dropout is not None:
            fc_1 = tf.nn.dropout(fc_1, dropout)
            
        out = tf.add(tf.matmul(fc_1, tf.get_variable('out')), tf.get_variable('out_bias'))
        
        return out
    


n_input = 784
n_labels = 10
learning_rate = 0.001
nb_epochs = 10
nb_samples_in_batch = 256

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_labels])

# weights and biases stored as dict

with tf.variable_scope('test', reuse=False):
    weights = {
        'conv1': tf.get_variable('conv1', initializer=tf.random_normal([5, 5, 1, 32])),
        'conv2': tf.get_variable('conv2', initializer=tf.random_normal([5, 5, 32, 64])),
        'fc1': tf.get_variable('fc1', initializer=tf.random_normal([7*7*64, 1024])),
        'out': tf.get_variable('out', initializer=tf.random_normal([1024, n_labels]))
        }

    biases = {
        'conv1': tf.get_variable('conv1_bias', initializer=tf.random_normal([32])),
        'conv2': tf.get_variable('conv2_bias', initializer=tf.random_normal([64])),
        'fc1': tf.get_variable('fc1_bias', initializer=tf.random_normal([1024])),
        'out': tf.get_variable('out_bias', initializer=tf.random_normal([n_labels]))
    }

# build CNN

sess = tf.InteractiveSession()
Y_pred = CNN(X, dropout=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred, Y))

optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

bool_accuracy = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(bool_accuracy, tf.float32))

nb_batches = int(mnist.train.num_examples / nb_samples_in_batch)

init = tf.initialize_all_variables()

#with tf.Session() as sess:
    
sess.run(init)

# loop over epochs
for i_epoch in range(nb_epochs):
    
    # loop over mini-batches
    for i_batch in range(nb_batches):
        
        # get next batch
        batch_x, batch_y = mnist.train.next_batch(nb_samples_in_batch)
        
        [_, cost_val, acc] = sess.run([optim, cost, accuracy], feed_dict={X: batch_x, Y:batch_y})
        
        print('epoch %i - batch %i - cost=%f - accuracy=%f' % (i_epoch, i_batch, cost_val, acc))
        
        
# test set
acc = sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y: mnist.labels[:256]})
print('Accuracy on test set: %f' % acc)
            



            
            
            
            
            
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    