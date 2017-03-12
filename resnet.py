# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:44:12 2017

@author: Florian

Deep residual networks to classify MNIST dataset

"""

# to be included in all python tensorflow files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

              
def conv_2D(x, w, b, stride=1, padding='SAME', activation=None):
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
    
    if activation is not None:
        x = activation(x)
    
    return x
    
def print_tensor_shape(x, msg=''):
    print(msg, x.get_shape().as_list())
              
              
class RepBlock(object):
    def __init__(self, num_repeats, num_filters, bottleneck_size, name_scope):
        self.num_repeats = num_repeats
        self.num_filters = num_filters
        self.bottleneck_size = bottleneck_size
        self.name_scope = name_scope
        
    def apply_block(self, net):
        
        print_tensor_shape(net, 'entering apply_block')
                
        # loop over repeats
        for i_repeat in range(self.num_repeats):
            
            print_tensor_shape(net, 'layer %i' % i_repeat)
            
            # subsampling is performed by a convolution with stride=2, only
            # for the first convolution of the first repetition
            if i_repeat == 0:
                stride = 2
            else:
                stride = 1
            
            name = self.name_scope+'/%i/conv_in' % i_repeat
            with tf.variable_scope(name):
                w = tf.get_variable('w', initializer=tf.truncated_normal([1, 1, net.get_shape().as_list()[-1], self.bottleneck_size], stddev=0.1))
                b = tf.get_variable('b', initializer=tf.truncated_normal([self.bottleneck_size], stddev=0.1))
                conv = conv_2D(net, w, b, stride=stride, padding='VALID', activation=tf.nn.relu)
                
            print_tensor_shape(conv, name)
                
            name = self.name_scope+'/%i/conv_bottleneck' % i_repeat    
            with tf.variable_scope(name):
                w = tf.get_variable('w', initializer=tf.truncated_normal([3, 3, conv.get_shape().as_list()[-1], self.bottleneck_size], stddev=0.1))
                b = tf.get_variable('b', initializer=tf.truncated_normal([self.bottleneck_size], stddev=0.1))
                conv = conv_2D(conv, w, b, stride=1, padding='SAME', activation=tf.nn.relu)
                
                print_tensor_shape(conv, name)
                
            name = self.name_scope+'/%i/conv_out' % i_repeat
            with tf.variable_scope(name):
                w = tf.get_variable('w', initializer=tf.truncated_normal([1, 1, conv.get_shape().as_list()[-1], self.num_filters], stddev=0.1))
                b = tf.get_variable('b', initializer=tf.truncated_normal([self.num_filters], stddev=0.1))
                conv = conv_2D(conv, w, b, stride=1, padding='VALID', activation=None)
                print_tensor_shape(conv, name)
                
            if i_repeat == 0:
                net = conv + tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            else:
                net = conv + net 
                
            net = tf.nn.relu(net)
            
            
        return net
                
            
            
            
            
            
        
        
              

def resnet(x):
    # reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # init block for each layer
    layer_1 = RepBlock(num_repeats=3, num_filters=128, bottleneck_size=32, name_scope='layer_1')
    layer_2 = RepBlock(num_repeats=3, num_filters=256, bottleneck_size=64, name_scope='layer_2')
#    layer_3 = RepBlock(num_repeats=3, num_filters=512, bottleneck_size=128, name_scope='layer_3')
#    layer_4 = RepBlock(num_repeats=3, num_filters=1024, bottleneck_size=256, name_scope='layer_4') 
    
    layers = [layer_1, layer_2]

    # first layer
    name = 'conv_1'
    with tf.variable_scope(name):
        w = tf.get_variable('w', initializer=tf.truncated_normal([7, 7, x.get_shape().as_list()[-1], 64], stddev=0.1))
        b = tf.get_variable('b', initializer=tf.truncated_normal([64], stddev=0.1))
        net = conv_2D(x, w, b, stride=1, padding='SAME', activation=tf.nn.relu)  
        
    print_tensor_shape(net)
        
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    print_tensor_shape(net)
    
    with tf.variable_scope('conv_2'):
        w = tf.get_variable('w', initializer=tf.truncated_normal([1, 1, net.get_shape().as_list()[-1], layers[0].num_filters], stddev=0.1))
        b = tf.get_variable('b', initializer=tf.truncated_normal([layers[0].num_filters], stddev=0.1))
        net = conv_2D(net, w, b, stride=1, padding='SAME', activation=tf.nn.relu)
        
        
    print_tensor_shape(net)
        
        
    for i_layer, layer in enumerate(layers):
        
        # pass the net through all blocks of the layer
        net = layer.apply_block(net)
        
        print_tensor_shape(net, 'After block')
        
        try:
            # upscale (depth) to the next block size
            next_block = layers[i_layer+1]
            with tf.variable_scope('upscale_%i' % i_layer):
                w = tf.get_variable('w', initializer=tf.truncated_normal([1, 1, net.get_shape().as_list()[-1], next_block.num_filters], stddev=0.1))
                b = tf.get_variable('b', initializer=tf.truncated_normal([next_block.num_filters], stddev=0.1))
                net = conv_2D(net, w, b, stride=1, padding='SAME', activation=tf.nn.relu)
        
            print_tensor_shape(net)
                
        except IndexError:
            pass
        
    # apply average pooling
    net = tf.nn.avg_pool(net, ksize=[1, net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1], 
                                     strides=[1, 1, 1, 1], padding='VALID')
                                     
    print_tensor_shape(net, msg='after average pooling')
    
    # fully connected layer
    with tf.variable_scope('fc'):
        w = tf.get_variable('w', initializer=tf.truncated_normal([256, 10], stddev=0.1))
        b = tf.get_variable('b', initializer=tf.truncated_normal([10], stddev=0.1))
        
        net = tf.reshape(net, shape=[-1, 256])
        net = tf.add(tf.matmul(net, w), b)
       
    print_tensor_shape(net, 'after fc')
    
    return net    
    
    

if __name__ == '__main__':
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    Y_pred = resnet(X)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
    optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    session = tf.InteractiveSession()
    init_op = tf.initialize_all_variables()
    session.run(init_op)
    
    nb_epochs = 2
    batch_size = 128
    training_size = mnist.train.num_examples
    
    nb_mini_batches = training_size // batch_size
    cumul_acc = 0    
    
    # loop over epochs    
    for i_epoch in range(nb_epochs):
        
        # loop over mini-batches
        for i_batch in range(nb_mini_batches):
            
            # get mini-batch
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            [_, cost_val, acc] = session.run([optim, cost, accuracy], feed_dict={X: batch_x, Y:batch_y})
            cumul_acc += acc 
            
            if i_batch % 10 == 0:
        
                print('epoch %i - batch %i - cost=%f - cumul_accuracy=%f' % (i_epoch, i_batch, cost_val, cumul_acc/10))
                cumul_acc = 0
            
            
    # test set
    acc = session.run(accuracy, feed_dict={X: mnist.test.images[:1024], Y: mnist.test.labels[:1024]})
    print('Accuracy on test set: %f' % acc)
            
            
        
        
    

    


        
        
    
    

        

