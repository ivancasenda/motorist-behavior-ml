import numpy as np 

from src.classifiers.linear_classifier import LinearClassifier
from src.loss.softmax import softmax_loss


class Softmax(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X, y=None, training=True):
        W = self.params['W']
        grads = {}

        scores = X.dot(W)
        if y is None:
            return scores

        loss, dX = softmax_loss(scores, y)

        loss += self.reg * np.sum(W * W)

        if training:
            dW = np.zeros_like(W)
            dW += X.T.dot(dX)
            dW += 2 * self.reg * W
            grads['W'] = dW

        return scores, loss, grads