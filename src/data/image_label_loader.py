import numpy as np 
import os 
import tensorflow as tf 


class ImageLabelLoader(object):
    def __init__(self, class_names, img_shape=(32, 32, 3)):
        """
        Construct a ImageLabelLoader object to load images and label with format (N, H, W, C)
        
        Inputs:
        - class_names: List of class names. 
        - img_shape: Image size and image channels.
        """
        self.class_names = class_names
        self.img_shape = img_shape


    def _get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        one_hot = tf.dtypes.cast(one_hot, tf.int32)
        # Integer encode the label
        return tf.argmax(one_hot)


    def _decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=self.img_shape[2])
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_shape[0], self.img_shape[1]])


    def load(self, file_path):
        """ 
        Load image and label from file path.

        Inputs:
        - file_path: image path structured with 'directory_path/class_name/img_file_name'
        """
        label = self._get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

