import unittest
from src.utils.data_util import train_val_split, train_val_split_unique, get_paths, sample_dataset
import numpy as np 
from math import ceil


class TestDataUtil(unittest.TestCase):
    """ Test for data_util module in utils. """

    def test_train_val_split(self):
        """ Unit test for train_val_split() """
        total_size = 80
        feature_dim = 9
        val_size = 0.3
        correct_val_size = ceil(total_size * val_size)

        X_arr = np.ones((total_size, feature_dim))
        y_arr = np.ones((total_size,))
        
        X_train, X_val, y_train, y_val = train_val_split(X_arr, y_arr, val_size=val_size, random_state=11)

        self.assertEqual(X_train.shape[0] + X_val.shape[0],  total_size)
        self.assertEqual(y_train.shape[0] + y_val.shape[0],  total_size)

        self.assertEqual(X_val.shape[0], correct_val_size)
        self.assertEqual(y_val.shape[0], correct_val_size)
        self.assertEqual(X_val.shape[1], feature_dim)
        self.assertEqual(X_train.shape[1], feature_dim)
        
        for i in range(3):
            X_train_new, X_val_new, y_train_new, y_val_new = train_val_split(X_arr, y_arr, val_size=val_size, random_state=11)
            self.assertTrue(np.all(X_train == X_train_new))


    def test_train_val_split_unique(self):
        """ Unit test for train_val_split_unique() """
        X_arr_uniqiue_1 = np.zeros((120, 15))
        X_arr_uniqiue_2 = np.ones((60, 15))
        X_arr = np.concatenate((X_arr_uniqiue_1, X_arr_uniqiue_2), axis=0)
        y_arr = np.zeros(180,)

        total_size = 180

        X_train, X_val, y_train, y_val = train_val_split_unique(X_arr, y_arr, column_split=0)

        self.assertEqual(X_train.shape[0] + X_val.shape[0],  total_size)
        self.assertEqual(y_train.shape[0] + y_val.shape[0],  total_size)

    
    def test_sample_dataset(self):
        total_size = 80
        feature_dim = 9
        dataset = np.ones((total_size, feature_dim))

        sampled_dataset = sample_dataset(dataset, num_sample=30)

        self.assertEqual(sampled_dataset.shape[0], 30)
        self.assertEqual(sampled_dataset.shape[1], feature_dim)


    def test_get_paths(self):
        base_dir = "test"
        file_names = ['img_01', 'img_02']
        label_names = ['c1', 'c2']

        paths = get_paths(base_dir, file_names, label_names)

        self.assertEqual(paths[0], 'test/c1/img_01')
        self.assertEqual(paths[1], 'test/c2/img_02')
