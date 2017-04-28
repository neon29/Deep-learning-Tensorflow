# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:22:54 2017

@author: Florian

Vanilla/fully connected conditional GAN for MNIST dataset

"""


from __future__ import division
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y):
    
    with tf.variable_scope('generator', reuse=True):
        
        G_W1 = tf.get_variable('G_W1')
        G_b1 = tf.get_variable('G_b1')

        G_W2 = tf.get_variable('G_W2')
        G_b2 = tf.get_variable('G_b2')
    
    z = tf.concat(values=[z, y], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x, y):
    
    with tf.variable_scope('discriminator', reuse=True):
        
        D_W1 = tf.get_variable('D_W1')
        D_b1 = tf.get_variable('D_b1')

        D_W2 = tf.get_variable('D_W2')
        D_b2 = tf.get_variable('D_b2')
        
    
    # concat x and y
    x = tf.concat(values=[x, y], axis=1)
        
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


if __name__ == '__main__':
    
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    
    folder = 'out_vanilla_conditional_GAN/'     
    
    # input size
    input_size = 784
    batch_size = 128
    # size conditional variable
    cond_size = 10
    # size of generator input
    Z_dim = 100 
    # batch within an epoch
    batches_per_epoch = int(np.floor(mnist.train.num_examples / batch_size))
    nb_epochs = 100
    
    learning_rate = 0.001
    
    # initialize weights and biases
    with tf.variable_scope('discriminator'):
        
        D_W1 = tf.get_variable('D_W1', initializer=xavier_init([input_size+cond_size, 128]))
        D_b1 = tf.get_variable('D_b1', initializer=tf.zeros(shape=[128]))

        D_W2 = tf.get_variable('D_W2', initializer=xavier_init([128, 1]))
        D_b2 = tf.get_variable('D_b2', initializer=tf.zeros(shape=[1]))
        
    theta_D = [D_W1, D_W2, D_b1, D_b2]
    
    with tf.variable_scope('generator'):
        
        G_W1 = tf.get_variable('G_W1', initializer=xavier_init([Z_dim+cond_size, 128]))
        G_b1 = tf.get_variable('G_b1', initializer=tf.zeros(shape=[128]))

        G_W2 = tf.get_variable('G_W2', initializer=xavier_init([128, input_size]))
        G_b2 = tf.get_variable('G_b2', initializer=tf.zeros(shape=[input_size]))
        
    theta_G = [G_W1, G_W2, G_b1, G_b2]
    
   
    
    
    X = tf.placeholder(tf.float32, shape=[None, input_size])
    
    Y = tf.placeholder(tf.float32, shape=[None, cond_size])

    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
    
    

        

    G_sample = generator(Z, Y)
    D_real, D_logit_real = discriminator(X, Y)
    D_fake, D_logit_fake = discriminator(G_sample, Y)
    
    
    # objective functions
    # discriminator aims at maximizing the probability of TRUE data (i.e. from the dataset) and minimizing the probability
    # of GENERATED/FAKE data:
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    
    # generator aims at maximizing the probability of GENERATED/FAKE data (i.e. fool the discriminator)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
    # when optimizing generator, discriminator is kept fixed
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)
    

    with tf.Session() as sess:    
    
        sess.run(tf.global_variables_initializer())
        
        if not os.path.exists(folder):
            print('CREATE')
            os.makedirs(folder)
            
        for i_epoch in range(nb_epochs):
            
            D_loss_cum = 0
            G_loss_cum = 0
            
            for i_batch in range(batches_per_epoch):
            
                X_mb, Y_mb = mnist.train.next_batch(batch_size)
                
                # train discriminator
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Y:Y_mb, Z: sample_Z(batch_size, Z_dim)})
                D_loss_cum += D_loss_curr
                
                # train generator
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Y:Y_mb, Z: sample_Z(batch_size, Z_dim)})
                G_loss_cum += G_loss_curr
        
            
            conditional_arr = np.zeros(shape=(16, cond_size))
            conditional_arr[:, 4] = 1
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim), Y:conditional_arr})
        
            fig = plot(samples)
            plt.savefig('%s/%i.png' % (folder, i_epoch), bbox_inches='tight')
            plt.close(fig)
        
            
            print('Iter: {}'.format(i_epoch))
            print('D loss: {:.4}'.format(D_loss_cum))
            print('G_loss: {:.4}'.format(G_loss_cum))
