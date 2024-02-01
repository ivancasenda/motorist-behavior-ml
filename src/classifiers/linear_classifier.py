from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange

class LinearClassifier(object):
    def __init__(self, input_dim, num_classes, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        
        self.params['W'] = weight_scale * np.random.randn(input_dim, num_classes)


    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        W = self.params['W']

        return X.dot(W)

    
    def loss(self, X_batch, y_batch=None):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - scores of each class
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


