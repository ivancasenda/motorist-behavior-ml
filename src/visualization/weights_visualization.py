import matplotlib.pyplot as plt 
import numpy as np 

def visualize_weights(weights, class_names, img_shape, bias_in_weights=True):
    plt.figure(figsize=(10, 5), dpi=100)
    num_classes = len(class_names)
    W = weights.copy()
    # Visualize the learned weights for each class.
    if bias_in_weights:
        W = W[:-1,:] # strip out the bias
    W = W.reshape(img_shape[0], img_shape[1], img_shape[2], num_classes)
    W_min, W_max = np.min(W), np.max(W)
    for i in range(num_classes):
        plt.subplot(2, 5, i + 1)
        
        # Rescale the weights to be between 0 and 255
        Wimg = 255.0 * (W[:, :, :, i].squeeze() - W_min) / (W_max - W_min)
        plt.imshow(Wimg.astype('uint8'))
        plt.axis('off')
        plt.title(class_names[i])