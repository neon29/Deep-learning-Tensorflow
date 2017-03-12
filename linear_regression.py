import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Full batch learning is used here
'''

if __name__ == '__main__':
    
    df = pd.read_csv('cars.csv')
    x_data, y_data = (df['speed'].values, df['dist'].values)
    
    
    nb_epochs = 200
    
#    x_data = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#    y_data = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    
    m = tf.Variable(1.0) # weights
    c = tf.Variable(1.0) # bias
    
    X = tf.placeholder(tf.float32, shape=(x_data.size)) 
    Y = tf.placeholder(tf.float32, shape=(y_data.size))
    
    Ypred = tf.add(tf.mul(X, m), c)
    
    init_op = tf.initialize_all_variables()
    
    loss = tf.reduce_mean(tf.squared_difference(Ypred, Y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    
    with tf.Session() as sess:
        
        sess.run(init_op)
        plt.figure()
        plt.plot(x_data, sess.run(Ypred, feed_dict={X:x_data}), label='Initial')
        plt.scatter(x_data, y_data )
        
        for epoch in range(nb_epochs):
            
            
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            
            print(sess.run(m), sess.run(c))
            
            plt.plot(x_data, sess.run(Ypred, feed_dict={X:x_data}))
            plt.legend()
        
       
   
    
    
        
        
    
    