import numpy as np 
import tensorflow as tf 

def tf_flatten(X):
    """    
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)
    
    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    N = tf.shape(X)[0]
    return tf.reshape(X, (N, -1))


def np_flatten(X):
    """    
    Input:
    - Numpy array of shape (N, D1, ..., DM)
    
    Output:
    - Numpy array of shape (N, D1 * ... * DM)
    """
    N = X.shape[0]
    return X.reshape(N, -1)


def bias_trick(X):
    """ Add bias dimension of ones to be used weight W that include bias """
    N = tf.shape(X)[0]
    return tf.concat([X, tf.ones([N, 1], tf.float32)], axis=1)