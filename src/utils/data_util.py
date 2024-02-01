import pandas as pd 
import numpy as np 
from math import ceil
from os import path


def sample_dataset(data, num_sample, random_state=None):
    """ 
    Sample dataset on axis 0, sample N in dataset with shape (N, ...)
    
    Inputs:
    - data: Data to be sampled in array. (N, ...)
    - num_sample: Size of sample. Must be less than data size.
    - random_state: Integer for reproducible sample.

    Returns:
        Sampled dataset (num_sample, ....).
    """
    data_size = len(data)
    if num_sample > data_size:
        raise ValueError('Number of sample is bigger than source data size')

    random = np.random.RandomState(random_state)
    permutation = random.permutation(data_size)

    sample_indices = permutation[:num_sample]

    return data[sample_indices]
    

def get_paths(base_dir, file_names, label_names):
    """ 
    Get concatenated list of 'base_dir/file_name/label_name'. 
    label_names[i] label for file_names[i].
    """
    return np.array([f'{base_dir}{path.sep}{label_name}{path.sep}{file_name}' for file_name, label_name in zip(file_names, label_names)])


def train_val_split(X, y, val_size=0.2, random_state=None):
    """
    Split data into train and validation set, preserve features dimension.

    Inputs:
    - X: (N, ...) numpy array of data to be split.
    - y: (N, ...) numpy array of label data to be split. y[i] gives the label for X[i]
    - val_size: Percentange of validation size to be split, between 0 - 1.
    - random_state: Controls the shuffling for reproducible split, int type.

    Returns:
    - X_train: Array of training set with size complement of validation size.
    - X_val: Array of validation set with specified size.
    - y_train: Array of training label, size same as X_train.
    - y_val: Array of validation label, size same as X_val.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Invalid input, data and label size doesn't match.")

    num_data = X.shape[0]
    num_val = ceil(val_size * num_data)
    num_train = num_data - num_val

    random = np.random.RandomState(random_state)
    permutation = random.permutation(num_data)

    val_indices = permutation[:num_val]
    train_indices = permutation[num_val:(num_train + num_val)]

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def train_val_split_unique(X, y, column_split, val_size=0.2, random_state=None):
    """
    Split data into train and validation set, The train and validation data are split 
    such that one column value on column_split can only appear on either train or validation set. 

    Inputs:
    - X: (N, ...) numpy array of data to be split.
    - y: (N, ...) numpy array of label data to be split. y[i] gives the label for X[i]
    - column_split: Column index where the column value can only appear on either train or validation set.
    - val_size: Percentange of validation size after split, between 0 - 1.
    - random_state: Controls the shuffling for reproducible split, int type.

    Returns:
    - X_train: Array of training set with value on column_split differ from validation.
    - X_val: Array of validation set, size doesn't stricly match specified val_size.
    - y_train: Array of training label, size same as X_train.
    - y_val: Array of validation label, size same as X_val.
    """
    num_data = X.shape[0]
    num_val = ceil(val_size * num_data)
    num_train = num_data - num_val

    value, count = np.unique(X[:, column_split], return_counts=True)

    random = np.random.RandomState(random_state)
    permutation = random.permutation(len(value))
    
    X_train = X.copy()
    y_train = y.copy()
    X_val = []
    y_val = []
    count_validation = 0
    for i in permutation:
        count_validation += count[i]
        value_index = np.where(X[:, column_split] == value[i])
        
        X_val.append(X[value_index])
        y_val.append(y[value_index])
        X_train = np.delete(X_train, value_index, axis=0)
        y_train = np.delete(y_train, value_index, axis=0)

        if count_validation >= num_val:
            break

    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    return X_train, X_val, y_train, y_val
