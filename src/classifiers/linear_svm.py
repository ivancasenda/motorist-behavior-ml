import numpy as np 

from src.classifiers.linear_classifier import LinearClassifier
from src.loss.linear_svm import svm_loss


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    
    def loss(self, X, y=None, training=True):
        """
        Multiclass support vector machines loss function, using hinge loss. vectorized implementation.
        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns:
        If y == None return scores
        else returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
        W = self.params['W']
        grads = {}

        scores = X.dot(W)
        if y is None:
            return scores

        loss, dX = svm_loss(scores, y)

        loss += self.reg * np.sum(W * W) # Add regularization

        if training:
            dW = np.zeros(W.shape) # Initialize the gradient as zero
            dW = X.T.dot(dX)
            dW += 2 * self.reg * W # Add regularization
            grads['W'] = dW

        return scores, loss, grads

        

