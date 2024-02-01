import tensorflow as tf
import pickle

class Standardizer(object):
    def __init__(self, eps=1e-8):
        """
        Construct a Standardizer to normalize data by minus mean and devide by std (x-mean/std).
        
        Inputs:
        - eps: Small scalar used for smoothing to avoid dividing by zero.
        """
        self.mean = None
        self.std = None
        self.eps = eps


    def fit(self, dataset, num_samples_batch=None, storepath="models/norm_param"):
        """
        Compute the mean and std to be used for normalizing.
        Compute on axis 0 on dataset with shape (N, ...)
        
        Inputs:
        - dataset: (tf.data.Dataset) Tensorflow dataset containing image, label
        """
        data = []
        for i, (X, _) in enumerate(dataset):
            data.append(X)

            if num_samples_batch != None:
                if i == num_samples_batch: break

        data = tf.concat(data, axis=0)

        self.mean = tf.reduce_mean(data, axis=0)
        self.std = tf.math.reduce_std(data, axis=0)

        norm_param = {
            "mean": self.mean,
            "std": self.std,
            'eps': self.eps
        }
        filename = "%s.pkl" % (storepath)
        with open(filename, "wb") as f:
            pickle.dump(norm_param, f)

    
    def transform(self, X):
        """ 
        Perform standardization by centering and scaling
        Inputs:
        - X: Data to be standardize

        Returns:
        - X: Standardized data
        """
        X = X - self.mean
        X = X / (self.std + self.eps)
        return X