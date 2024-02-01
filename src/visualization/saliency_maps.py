import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """

    N = X.shape[0]
    X_var = tf.convert_to_tensor(X)
    y_var = tf.convert_to_tensor(y, dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(X_var)
        scores = model(X_var, training=False)
        correct_scores = tf.gather_nd(scores, tf.stack((tf.range(N), y_var), axis=1))
        grads = tape.gradient(correct_scores, X_var)
        
    saliency = np.abs(grads)
    saliency = np.max(saliency, axis=3)

    return saliency


def show_saliency_maps(model, X, y, X_img, mask, class_names):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    saliency = compute_saliency_maps(Xm, ym, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(X_img[i])
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()