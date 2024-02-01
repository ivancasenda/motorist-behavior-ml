import numpy as np 


def log_loss(y_true, y_scores, scores_in_prob=False):
    """ Calculate kaggle log loss """
    num_pred = y_scores.shape[0]
    
    if scores_in_prob:
        probabilities = y_scores
    else:
        scores = np.exp(y_scores) # normalized
        scores_sum = np.sum(scores, axis=1).reshape(-1, 1)
        probabilities = scores / scores_sum

    log_probabilities = np.log(probabilities)

    log_pred = log_probabilities[np.arange(num_pred), y_true]

    return -np.sum(log_pred)/num_pred


def accuracy(y_true, y_pred):
    """ Calculate accuracy """
    num_correct = np.sum(y_pred == y_true)
    num_pred = len(y_pred)
    accuracy = float(num_correct) / num_pred
    return accuracy


def confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix.
    Labels obtain through unique value in y_true.

    Inputs:
    - y_true: Numpy array of actual target values.
    - y_pred: Numpy array of predicted values.

    Returns:
     - confusion_matrix: with i-th row indicates the true class
                        and j-th column indicates the predicted class.
    """
    num_labels = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_labels, num_labels))
    
    for true_class, pred_class in zip(y_true, y_pred):
        confusion_matrix[true_class][pred_class] += 1

    return confusion_matrix 