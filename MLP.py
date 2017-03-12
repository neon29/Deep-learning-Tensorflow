# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:27:23 2017

@author: Florian

Implementation of multi-layer perceptron

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# model parameters
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_label = 10
learning_rate = 0.001

batch_size = 256
training_epochs = 15



def MLP(X):
    with tf.variable_scope('mlp', reuse=False):
        w1 = tf.get_variable('w1', initializer=tf.random_uniform([n_input, n_hidden_1], -1, 1))
        b1 = tf.get_variable('b1', initializer=tf.random_uniform([n_hidden_1], -1, 1))
        w2 = tf.get_variable('w2', initializer=tf.random_uniform([n_hidden_1, n_hidden_2], -1, 1))
        b2 = tf.get_variable('b2', initializer=tf.random_uniform([n_hidden_2], -1, 1))
        w_out = tf.get_variable('w_out', initializer=tf.random_uniform([n_hidden_2, n_label], -1, 1))
        b_out = tf.get_variable('b_out', initializer=tf.random_uniform([n_label], -1, 1))

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, w1), b1))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    layer_out = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w_out), b_out))

    return layer_out

# Create model
def MLP2(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_label]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_label]))
}
    
    
    
    
X = tf.placeholder('float', shape=[None, n_input])
Y = tf.placeholder('float', shape=[None, n_label])

y_pred = MLP(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=Y))

optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)

    nb_batches = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epochs):
        
        avg_cost = 0

        for i in range(nb_batches):

            # get batch
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([optim, cost], feed_dict={X: batch_x, Y: batch_y})
            
            avg_cost = avg_cost + cost_val

            
        print('epoch %i, averaged cost=%f' % (epoch, avg_cost/batch_size))
        
    # Test model
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))




    
