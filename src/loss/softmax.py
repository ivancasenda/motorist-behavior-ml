from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log

def softmax_loss(scores, y):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = scores.shape[0]
        
    scores = np.exp(scores) # normalized
    scores_sum = np.sum(scores, axis=1).reshape(-1, 1)
    prob_out = scores / scores_sum

    dX = prob_out.copy()
    dX[np.arange(num_train), y] -= 1 # prob_out - true_prob
    dX /= num_train

    loss = prob_out[np.arange(num_train), y]
    loss = np.sum(-np.log(loss))
    loss /= num_train

    return loss, dX