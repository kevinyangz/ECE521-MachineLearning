import tensorflow as tf
import numpy as np

def euclideanDistanceMultipleDimension(TrainData,TestData):
    squareTrain = tf.reduce_sum(TrainData*TrainData, 1)
    squareTest = tf.reduce_sum(TestData*TestData,1)
    squareTrain= tf.reshape(squareTrain, [-1, 1])
    squareTest= tf.reshape(squareTest, [-1, 1])
    euclideanMatrix = squareTrain - 2*tf.matmul(TrainData, tf.transpose(TestData)) + tf.transpose(squareTest)
    return euclideanMatrix

def euclideanV3(x,z):
    #get the dimension of z,x 
    z_dim = tf.shape(z) #z_dim[0] = num of rows, z_dim[1] = num of columns
    z_row = z_dim[0]
    z_col = z_dim[1]
    x_dim = tf.shape(x) 
    x_row = x_dim[0]
    x_col = x_dim[1]
    
    #duplicate each row in x by z_row times
    x_expanded = tf.tile(x,[1,z_row],name='x_expanded') #note this is a flattend array
    #print (sess.run(x_expanded))
    # get the desired shape 
    x_reshaped = tf.reshape(x_expanded,[x_row,z_row,z_col])
    #print (sess.run(x_reshaped))
    
    #get the square of the difference, tensorflow broadcasts z to match the size of s
    square_difference = tf.square(x_reshaped-z)
    #reduce sum each row
    #print (sess.run(square_difference))
    d = tf.reduce_sum(square_difference,2)
    return d

