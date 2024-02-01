import unittest
from src.utils.data_util import get_paths, sample_dataset
from src.data.dataset_loader import load_dataset_image_label, load_distracted_driver_detection_list
import numpy as np 
import pandas as pd
from math import ceil


class TestLoadData(unittest.TestCase):
    """ Test for load_data module in data. """

    def test_load_dataset_image_label(self):
        TRAIN_IMG_DIRECTORY = "dataset/raw/imgs/train"
        X_train_filenames, X_val_filenames, y_train_labels, y_val_labels = load_distracted_driver_detection_list(val_size=0.2, split_on_driver=False, random_state=11)
        train_paths = get_paths(TRAIN_IMG_DIRECTORY, X_train_filenames, y_train_labels)
        train_paths = sample_dataset(train_paths, num_sample=500)

        num_train = len(train_paths)
        img_size = 256

        X_train, y_train = load_dataset_image_label(train_paths, img_size=img_size)
        self.assertEqual(X_train.shape[2], img_size)
        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(len(y_train), num_train)

        val_paths = get_paths(TRAIN_IMG_DIRECTORY, X_val_filenames, y_val_labels)
        val_paths = sample_dataset(val_paths, num_sample=500)

        num_val = len(val_paths)
        img_size = 256

        X_val, y_val = load_dataset_image_label(val_paths, img_size=img_size)
        self.assertEqual(X_val.shape[2], img_size)
        self.assertEqual(X_val.shape[0], num_val)
        self.assertEqual(len(y_val), num_val)

    
    def test_load_distracted_driver_detection_list(self):
        X_train_filenames, X_val_filenames, y_train_labels, y_val_labels = load_distracted_driver_detection_list(val_size=0.2, split_on_driver=False, random_state=11)
        self.assertEqual(X_train_filenames.shape[0], y_train_labels.shape[0])
        self.assertEqual(X_val_filenames.shape[0], y_val_labels.shape[0])

        X_train_filenames, X_val_filenames, y_train_labels, y_val_labels = load_distracted_driver_detection_list(val_size=0.2, split_on_driver=True, random_state=11)
        self.assertEqual(X_train_filenames.shape[0], y_train_labels.shape[0])
        self.assertEqual(X_val_filenames.shape[0], y_val_labels.shape[0])