import unittest
from src.preprocessing.batch import tf_flatten, np_flatten, bias_trick
import numpy as np 
#import tensorflow as tf 


class TestBatch(unittest.TestCase):
    """ Test for image module in preprocessing. """

    def test_tf_flatten(self):
        x_np = np.arange(24).reshape((2, 3, 4))
        x_flat_np = tf_flatten(x_np)
        
        self.assertEqual(x_flat_np.shape[0], x_np.shape[0])
        self.assertEqual(x_flat_np.shape[1], 12)

    
    def test_np_flatten(self):
        x_np = np.arange(24).reshape((2, 3, 4))
        x_flat_np = np_flatten(x_np)
        
        self.assertEqual(x_flat_np.shape[0], x_np.shape[0])
        self.assertEqual(x_flat_np.shape[1], 12)

    
    def test_bias_trick(self):
        x_np = np.arange(24).reshape((4, 6))
        x_np = bias_trick(x_np)

        self.assertEqual(x_np.shape[0], 4)
        self.assertEqual(x_np.shape[1], 7)
        self.assertEqual(x_np[0, 6], 1)

