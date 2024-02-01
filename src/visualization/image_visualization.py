import matplotlib.pyplot as plt
import numpy as np 


def visualize_train_images(class_names, X_train, y_train, samples_per_class=7):
    """
    Visualize some examples from the dataset.
    We show a few examples of training images from each class.
    
    Inputs:
    - class_names: List of class name in the image dataset.
    - X_train: Images in numpy array.
    - y_train: Label of X_train. y_train[i] label for X_train[i]
    - samples_per_class: Number of samples to be displayed per class.
    """
    plt.figure(figsize=(10, 5), dpi=150)
    num_classes = len(class_names)
    for y, cls in enumerate(class_names):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=True)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx])
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


