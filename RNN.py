# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:12:30 2017

@author: Florian

Implement recurrent neural network (LSTM) to classify digits
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_steps = 28
n_inputs = 28
n_classes = 10

batch_size = 128
n_epochs = 1000
n_hidden = 256

learning_rate = 0.001

display_step = 10

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_classes])


# set weights and biases
RNN_params = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])), 
              'biases':tf.Variable(tf.random_normal([n_classes]))}
              
# build 
#def RNN(x, weights):
    
# input tensor shape is (batch_size, n_steps, n_inputs) but need
# to be n_steps tensor of shape (batch_size, n_inputs), so reshape this

# (n_steps, batch_size, n_inputs)
X1 = tf.transpose(X, [1, 0, 2])

# (batch_size*n_steps, n_inputs)
X2 = tf.reshape(X1, [-1, n_inputs])

# list of n_steps tensors of shape (batch_size, n_inputs)
X3 = tf.split(0, n_steps, X2)

# get lstm cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Get lstm cell output
outputs, states = tf.nn.rnn(lstm_cell, X3, dtype=tf.float32)

print(len(outputs))
print(outputs[0].get_shape())
print(len(states))
print(states[0].get_shape())

# use last output of the rnn as output
#return tf.matmul(outputs[-1], RNN_params['weights']) + RNN_params['biases']
print(outputs[0].get_shape())
print(RNN_params['weights'].get_shape())
print(RNN_params['biases'].get_shape())


Y_pred = tf.matmul(outputs[-1], RNN_params['weights']) + RNN_params['biases']
    
sess = tf.InteractiveSession()
    


#Y_pred = RNN(X, RNN_params)

# define the loss and the optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
              
correct_pred = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.initialize_all_variables()

# Launch the graph
#with tf.Session() as sess:
sess.run(init_op)
step = 1
# Keep training until reach max iterations
while step * batch_size < n_epochs:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, n_steps, n_inputs))
    # Run optimization op (backprop)
    sess.run(optim, feed_dict={X: batch_x, Y: batch_y})
    
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
        # Calculate batch loss
        loss_val = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss_val) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_inputs))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    