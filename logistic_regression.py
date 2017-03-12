# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:27:37 2017

@author: Florian

Logistic regression for supervised learning using the MNIST dataset

"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.02
training_epochs = 100 # number of times we process the full dataset
batch_size = 10 # number of examples to use to perform a single parameters
# update step

# placeholders to pass inputs (X) and true labels (Y) during training.
# when passing inputs, their tensor size will be (nb_examples, 27*27=784 pixels)
X = tf.placeholder(tf.float32, shape=(None, 784)) 

# the output shape will thus be (nb_examples, 10) as there are 10 possible labels
Y = tf.placeholder(tf.float32, shape=(None, 10))

# to obtain a tensor of shape (1, 10) at the ouptut from the input of shape
# (1, 784) we need a weights matrix of shape (784, 10) and a bias term whose
# shape is (1, 10):

W = tf.get_variable('weights', dtype=tf.float32, initializer=tf.zeros_initializer(shape=(784, 10)))
b = tf.get_variable('bias', dtype=tf.float32, initializer=tf.zeros_initializer(shape=(1, 10)))

# softmax
pred = tf.nn.softmax(tf.matmul(X, W) + b)

'''
we aim at minimizing cross-entropy:
# 1: tf.reduce_sum(..., axis=1): for each example in the current batch, it 
# sums y*log(pred) over the 10 labels (axis=1 as pred is of shape 
(nb_examples_in_batch, 10)), the output is thus of shape (nb_examples_in_batch,)
# 2: we compute the mean of cross-entropies of all examples within the batch
    
'''
cost = - tf.reduce_mean(tf.reduce_sum(Y*tf.log(pred), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.initialize_all_variables()


with tf.Session() as session:
    
    session.run(init_op)
    
    nb_batches = int(mnist.train.num_examples/batch_size)
    
    # loop over training epochs
    for epoch in range(training_epochs):
        
        averaged_cost = 0
        
        # loop over all batches
        for batch in range(nb_batches):
            
            # get training examples
            x_s, y_s = mnist.train.next_batch(batch_size)
            
            # run optimizer
            _, current_cost = session.run([optimizer, cost], feed_dict={X: x_s, Y: y_s})
            
            averaged_cost = averaged_cost + current_cost / nb_batches
            
        print('epoch %i (averaged cost = %f)' % (epoch, averaged_cost))    
        
        
        
        # evaluate on test set at each epoch:
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        print('accuracy = %f' % accuracy.eval(feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
        
    
    
    
    

