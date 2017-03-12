# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:30:10 2017

@author: Florian
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


session = tf.InteractiveSession()

nb_neighbors = 5

# training
Xtr, Ytr = mnist.train.next_batch(5000) 
# test
Xte, Yte = mnist.test.next_batch(200) 


# placeholder for training
xtr = tf.placeholder("float", [None, 784])
ytr = tf.placeholder('float', [None, 10])

# placeholder for testing
xte = tf.placeholder("float", [784])

# to compute nearest neighbors we can choose any norm (L2 here)
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), axis=1))

# get the index of the 'nb_neighbors' nearest neighbors
values, indices = tf.nn.top_k(-distance, k=nb_neighbors)

pred = tf.argmax(tf.reduce_sum(tf.gather(ytr, indices), axis=0), axis=0)

accuracy = 0.0


init = tf.initialize_all_variables()

with tf.Session() as session:  
    
    for i in range(len(Xte)):
        
        #nn_index = session.run(indices, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        predicted_class = pred.eval(feed_dict={xtr: Xtr, xte: Xte[i, :], ytr: Ytr})
        
        print("Test", i, "Prediction:", predicted_class, \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if predicted_class == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)