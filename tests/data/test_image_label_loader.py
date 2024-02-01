import unittest
from src.utils.data_util import get_paths
from src.data.dataset_loader import load_distracted_driver_detection_list
from src.data.image_label_loader import ImageLabelLoader
import pathlib
import numpy as np 
import tensorflow as tf 


class TestImageLoader(unittest.TestCase):
    """ Test for load_data module in data. """

    def test_image_loader(self):
        data_dir = pathlib.Path("dataset/raw/imgs/train") # Train directory

        X_train_filenames, X_val_filenames, y_train_labels, y_val_labels = load_distracted_driver_detection_list(val_size=0.2, split_on_driver=True, random_state=12)
        train_paths = get_paths(data_dir, X_train_filenames, y_train_labels)
        val_paths = get_paths(data_dir, X_val_filenames, y_val_labels)
        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

        CPU = '/cpu:0'
        #GPU = '/device:GPU:0'

        batch_size = 32 # Load all to 1 batch
        img_size = 256
        loader = ImageLabelLoader(class_names, img_size)

        # Force image load and preprocessing with specific device
        with tf.device(CPU):
            # Train dataset input pipeline
            train_dset = tf.data.Dataset.from_tensor_slices(train_paths)
            train_dset = train_dset.map(loader.load, num_parallel_calls=tf.data.experimental.AUTOTUNE) # Load from path to image, label
            train_dset = train_dset.batch(batch_size, drop_remainder=False)
            train_dset = train_dset.cache()
            train_dset = train_dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            # Validation dataset input pipeline
            val_dset = tf.data.Dataset.from_tensor_slices(val_paths)
            val_dset = val_dset.map(loader.load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_dset = val_dset.batch(batch_size, drop_remainder=False)
            val_dset = val_dset.cache()
            val_dset = val_dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


        X_train_batch, y_train_batch = next(iter(train_dset))
        X_train_batch = X_train_batch.numpy()
        y_train_batch = y_train_batch.numpy()
        
        self.assertEqual(X_train_batch.shape[0], batch_size)
        self.assertEqual(y_train_batch.shape[0], batch_size)

        self.assertEqual(X_train_batch.shape[2], img_size)

        X_val_batch, y_val_batch = next(iter(val_dset))
        X_val_batch = X_val_batch.numpy()
        y_val_batch = y_val_batch.numpy()

        self.assertEqual(X_val_batch.shape[0], batch_size)
        self.assertEqual(y_val_batch.shape[0], batch_size)

        self.assertEqual(X_val_batch.shape[2], img_size)