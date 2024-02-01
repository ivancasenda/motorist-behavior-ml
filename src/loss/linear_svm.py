from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss(scores, y):
    num_train = scores.shape[0]
    loss = 0.0

    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)

    #margins = scores - correct_class_scores + 1
    #margins[margins < 0] = 0
    #loss += (np.sum(margins) - num_train)
    
    margins = np.maximum(0, scores - correct_class_scores + 1) # Hinge loss
    margins[np.arange(num_train), y] = 0 
    loss += np.sum(margins)
    loss /= num_train

    # Margin mask for derivative, zero derivative for x == 0
    valid_margin_mask = np.zeros(margins.shape)
    valid_margin_mask[margins > 0] = 1
    valid_margin_mask[np.arange(num_train), y] = -np.sum(valid_margin_mask, axis=1)

    dX = valid_margin_mask / num_train

    return loss, dX