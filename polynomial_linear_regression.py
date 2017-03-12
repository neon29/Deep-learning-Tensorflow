import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Mini batch learning with a single example is used here
'''

if __name__ == '__main__':
    
    #df = pd.read_csv('cars.csv')
    #x_data, y_data = (df['speed'].values, df['dist'].values)
    
    plt.close('all')
    
    
    nb_epochs = 200
    N_deg = 1
    learning_rate = 0.0001
    
#    x_data = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#                         
#    ind_x = np.argsort(x_data)  
#    x_data = x_data[ind_x]                   
#                         
#    y_data = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#    y_data = y_data[ind_x]           
      
    n_observations = 100
    x_data = np.linspace(-3, 3, n_observations)
    y_data = np.sin(x_data) #+ np.random.uniform(-0.5, 0.5, n_observations)                     
                         
    
    
    #m = tf.Variable(1.0) # weights
    #c = tf.Variable(1.0) # bias
    
    Ypred = tf.Variable(1.0, name='bias')
    
    X = tf.placeholder(tf.float32) 
    Y = tf.placeholder(tf.float32)
    
    # let's use a polynomial regression up to degree N_deg
    with tf.variable_scope('test'):
        for i in range(1, N_deg+1):
                # weight for the current degree
                W = tf.get_variable(name='W_%i' % i, shape=(1), dtype=tf.float32)
                Ypred = tf.add(tf.mul(tf.pow(X, i), W), Ypred)
    
    
    init_op = tf.initialize_all_variables()
    
    loss = tf.reduce_mean(tf.squared_difference(Ypred, Y))
    #loss = tf.reduce_sum(tf.pow(Ypred - Y, 2)) / (n_observations - 1)

    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
    with tf.variable_scope('test', reuse=True):
        with tf.Session() as sess:
        
        
            #print(sess.run(tf.get_variable('W_1', shape=1)))
            
            sess.run(init_op)
            plt.figure()
            plt.plot(x_data, Ypred.eval(
                    feed_dict={X: x_data}, session=sess), label='initial')
            plt.scatter(x_data, y_data, color='k' )
            
            for epoch in range(nb_epochs):
                
                for (x, y) in zip(x_data, y_data):
                
                    sess.run(optimizer, feed_dict={X: x, Y: y})
                    
                    
                    
                print('Iter %i: %f' %(epoch, sess.run(loss, feed_dict={X: x_data, Y: y_data})))
                
                
                #plt.plot(x_data, sess.run(Ypred, feed_dict={X:x_data}))
                print(tf.get_variable('W_1').eval())
                plt.plot(x_data, Ypred.eval(
                    feed_dict={X: x_data}, session=sess))
            
            plt.legend()
        
       
   
    
    
        
        
    
    