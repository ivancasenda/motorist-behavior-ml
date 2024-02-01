import unittest
import numpy as np 
from src.evaluation.metrics import confusion_matrix
from src.utils.time_measure import time_function
from sklearn.metrics import confusion_matrix as sk_conf_matrix


class TestMetrics(unittest.TestCase):
    """ Test for metrics module in evalutaion. """

    def test_confusion_matrix(self):
        """ Unit test for confusion_matrix() """
        num_labels = 10
        size = 1000
        y_true = np.random.randint(0, num_labels, size)
        y_pred = y_true.copy()

        conf_matrix = confusion_matrix(y_true, y_pred)

        self.assertEqual(conf_matrix.shape[0], num_labels)
        
        zero_class_total = np.sum(y_true == 0)
        self.assertEqual(np.sum(conf_matrix[0]), zero_class_total)

        #print("\n")
        #conf_matrix_time = time_function(confusion_matrix, y_true, y_pred)
        #print(conf_matrix_time)
        #sk_conf_matrix_time = time_function(sk_conf_matrix, y_true, y_pred)
        #print(sk_conf_matrix_time)
