import pandas as pd 
import numpy as np 
import os 
import pathlib
from skimage.io import imread
from skimage.transform import resize

from src.utils.data_util import train_val_split, train_val_split_unique


def load_dataset_image_label(img_paths, img_size=32):
    """ 
    Load train images and labels to numpy array.
    Assumes images in directory structured like this example
    - c01 (directory with class name)
        - img_0001.jpg (image with filename)
        - img_00223.jpg
    - c02 
        - img_330.jpg
        - img_1189.jpg
    ....
    
    Inputs:
        - img_paths: List of image paths. structured with "directory_path/class_name/img_file_name"
        - img_size: Image size to be resized.

    Returns:
    - images: Images in numpy array with shape (N, H, W, C)
                N: Number of images in array.
                H: Image height.
                W: Image width.
                C: Image channels.
    - labels: Integer labels in numpy array, labels[i] label for images[i].
    """
    # Get class names
    dir_parts = img_paths[0].split(os.path.sep)
    data_dir = pathlib.Path('/'.join([part for part in dir_parts[:-2]]))
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

    images = []
    labels = []
    for path in img_paths:
        image = resize(imread(path), (img_size, img_size, 3), anti_aliasing=False)
        images.append(image)

        # convert the path to a list of path components
        parts = path.split(os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        labels.append(np.argmax(one_hot))

    return np.array(images), np.array(labels)


def load_distracted_driver_detection_list(val_size=0.2, split_on_driver=False, random_state=None):
    """
    Split distracted driver image list into train and validation list

    Inputs:
    - val_size: Percentange of validation size after split, between 0 - 1.
    - split_on_driver: boolean, if true The train and validation data are split such that one 
                        column value on column_split can only appear on either train or validation set.

    Returns:
     - X_train_filenames: List of train image filenames. If split_on_driver True, 
                            doesn't include image filenames whose driver appear in validation.
     - X_val_filenames: List of validation image filenames with specified percentage.
     - y_train_label: List of train label names in string. y_train_label[i] label for X_train_filenames[i].
     - y_val_label: List of validation label names in string. y_val_label[i] label for X_val_filenames[i].
    """
    RAW_DATASET_CSV = "dataset/raw/driver_imgs_list.csv"
    DRIVER_ID_COLUMN = 0
    LABEL_COLUMN = 1
    FILENAME_COLUMN = 2

    # Read dataset csv, list of driver_id, label, img filenames
    driver_imgs_list = pd.read_csv(RAW_DATASET_CSV).values
    X = driver_imgs_list[:, [DRIVER_ID_COLUMN, FILENAME_COLUMN]]
    y = driver_imgs_list[:, LABEL_COLUMN]

    if split_on_driver:
        X_train_filenames, X_val_filenames, y_train_label, y_val_label = train_val_split_unique(X, y, val_size=val_size, column_split=DRIVER_ID_COLUMN, random_state=random_state)
    else:
        X_train_filenames, X_val_filenames, y_train_label, y_val_label = train_val_split(X, y, val_size=val_size, random_state=random_state)

    X_train_filenames = X_train_filenames[:,1]
    X_val_filenames = X_val_filenames[:,1]

    return X_train_filenames, X_val_filenames, y_train_label, y_val_label


