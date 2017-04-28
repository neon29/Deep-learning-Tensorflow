# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:22:54 2017

@author: Florian

Conditional DCGAN for MNIST dataset

"""

from __future__ import division
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os



    
def leaky_relu(x, alpha):
    return tf.maximum(alpha * x, x)
    

    

def discriminator(x, y, is_training):
    
    with tf.variable_scope('discriminator', reuse=True):
        
        D_w_1 = tf.get_variable('D_w_1')
        D_w_2 = tf.get_variable('D_w_2')
        D_w_fc_1 = tf.get_variable('D_w_fc_1')
    
    # conv_2D accepts shape (batch, height, width, channel) as input so
    # reshape it
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    out = tf.nn.conv2d(x, D_w_1, strides=[1, 2, 2, 1], padding='SAME') 
    
    print(out.get_shape().as_list())
    # batch normalization
    out = tf.reshape(out, [-1, 14, 14, 64])
    out = tf.contrib.layers.batch_norm(out, is_training=is_training)
    
    out = leaky_relu(out, alpha=0.2)
    #out = tf.nn.dropout(out, keep_prob=0.2)
    out = tf.nn.conv2d(out, D_w_2, strides=[1, 2, 2, 1], padding='SAME') 
    
    print(out.get_shape().as_list())

    # batch normalization
    out = tf.reshape(out, [-1, 7, 7, 128])
    out = tf.contrib.layers.batch_norm(out, is_training=is_training)
    
    out = leaky_relu(out, alpha=0.2)
    #out = tf.nn.dropout(out, keep_prob=0.2)
    
    # fully connected layer
    out = tf.reshape(out, shape=[-1, 7*7*128])
    
    # concatenate the first fully connected layer with the conditional 
    # information
    out = tf.concat(values=[out, y], axis=1)
    
    D_logits = tf.matmul(out, D_w_fc_1)
    
    print(out.get_shape().as_list())

    # batch normalization
    out = tf.reshape(out, [-1, 7*7*128])
    out = tf.contrib.layers.batch_norm(out, is_training=is_training)
    
    #D_logits = tf.nn.sigmoid(D_logits)
    D_logits = leaky_relu(D_logits, alpha=0.2)
    
    return D_logits
    
    


def generator(z, y, is_training):
    
    with tf.variable_scope('generator', reuse=True):
        
        G_w_fc_1 = tf.get_variable('G_w_fc_1')
        G_w_deconv_1 = tf.get_variable('G_w_deconv_1')
        G_w_deconv_2 = tf.get_variable('G_w_deconv_2')
        
    # concatenante random input with conditional information
    z = tf.concat(values=[z, y], axis=1)    
    out = tf.matmul(z, G_w_fc_1)
    
    print(out.get_shape().as_list())

    # batch normalization
    out = tf.reshape(out, shape=[-1, 6272])
    out = tf.contrib.layers.batch_norm(out, is_training=is_training, reuse=None)
    
    out = tf.nn.relu(out)
    
    
    out = tf.reshape(out, shape=[-1, 7, 7, 128])

    out = tf.nn.conv2d_transpose(out, 
                                 G_w_deconv_1, 
                                 output_shape=tf.stack([tf.shape(out)[0], 14, 14, 64]),
                                 strides=[1, 2, 2, 1],
                                 padding='SAME') 
                                 
    
            
                     
    print(out.get_shape().as_list())

    # batch normalization
    out = tf.reshape(out, [-1, 14, 14, 64])
    out = tf.contrib.layers.batch_norm(out, is_training=is_training, reuse=None)      

                              
    out = tf.nn.relu(out)
    out = tf.nn.conv2d_transpose(out, 
                                 G_w_deconv_2, 
                                 output_shape=tf.stack([tf.shape(out)[0], 28, 28, 1]),
                                 strides=[1, 2, 2, 1],
                                 padding='SAME') 
                                 
                                 
    out = tf.nn.tanh(out)
    
        
    return out
    
    





def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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
    
    folder = 'out_conditional_DCGAN/'
    
    batch_size = 128
    # input size
    input_size = 784
    # size of generator input
    Z_dim = 10 
    # size of conditional information (10 for one-hot MNIST)
    cond_size = 10
    # batch within an epoch
    batches_per_epoch = int(np.floor(mnist.train.num_examples / batch_size))
    nb_epochs = 20
    
    # learning rate
    learning_rate = 0.00005 # 0.0002
    
    # label to generate when generating samples
    label_to_sample = np.zeros(shape=(16, cond_size))
    label_to_sample[:, 8] = 1
    
    Z = tf.placeholder(tf.float32, [None, Z_dim])
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, cond_size])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('discriminator'):
        D_w_1 = tf.get_variable('D_w_1', initializer=tf.random_normal([5, 5, 1, 64], stddev=0.02))
        D_w_2 = tf.get_variable('D_w_2', initializer=tf.random_normal([5, 5, 64, 128], stddev=0.02))
        D_w_fc_1 = tf.get_variable('D_w_fc_1', initializer=tf.random_normal([7*7*128+cond_size, 1], stddev=0.02)) 
        
    D_var_list = [D_w_1, D_w_2, D_w_fc_1]

        
    with tf.variable_scope('generator'):
        G_w_fc_1 = tf.get_variable('G_w_fc_1', initializer=tf.random_normal([Z_dim+cond_size, 128*7*7], stddev=0.02))
        G_w_deconv_1 = tf.get_variable('G_w_deconv_1', initializer=tf.random_normal([5, 5, 64, 128], stddev=0.02))
        G_w_deconv_2 = tf.get_variable('G_w_deconv_2', initializer=tf.random_normal([5, 5, 1, 64], stddev=0.02))
        
    G_var_list = [G_w_fc_1, G_w_deconv_1, G_w_deconv_2]
        

    G_sample = generator(Z, Y, is_training)
    D_logit_real = discriminator(X, Y, is_training)
    D_logit_fake = discriminator(G_sample, Y, is_training)
    
    
    # objective functions
    # discriminator aims at maximizing the probability of TRUE data (i.e. from the dataset) and minimizing the probability
    # of GENERATED/FAKE data:
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    
    # generator aims at maximizing the probability of GENERATED/FAKE data (i.e. fool the discriminator)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=D_var_list)
    # when optimizing generator, discriminator is kept fixed
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=G_var_list)
    

    with tf.Session() as sess:    
    
        sess.run(tf.global_variables_initializer())
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        for i_epoch in range(nb_epochs):
            
            G_loss_val = 0
            D_loss_val = 0
            
            for i_batch in range(batches_per_epoch):
                print('batch %i/%i' % (i_batch+1, batches_per_epoch))
            
                X_mb, Y_mb = mnist.train.next_batch(batch_size)
                
                # train discriminator
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Y:Y_mb, is_training:True, Z: sample_Z(batch_size, Z_dim)})
                D_loss_val += D_loss_curr
                
                # train generator
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim), Y:Y_mb, is_training:True})
                G_loss_val += G_loss_curr
                
                if i_batch % 50 == 0:
                    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim), Y:label_to_sample, is_training:False})
        
                    fig = plot(samples)
                    plt.savefig('%s/%i_%i.png' % (folder, i_epoch, i_batch), bbox_inches='tight')
                    plt.close(fig)
        
        
            
        
            
            print('Iter: {}'.format(i_epoch))
            print('D loss: {:.4}'.format(D_loss))
            print('G_loss: {:.4}'.format(G_loss))
        
        
       

