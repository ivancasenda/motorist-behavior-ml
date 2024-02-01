import matplotlib.pyplot as plt 
import numpy as np


def plot_loss_acc_history(loss_history, train_acc_history, val_acc_history):
    # Visualize training loss and train / val accuracy
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(10, 9)
    plt.show()


def plot_loss_acc_history_epoch(loss_history, val_loss_history, train_acc_history, val_acc_history):
    # Visualize training loss and train / val accuracy
    x_epoch = np.arange(1, len(loss_history)+1, step=1)

    plt.subplot(2, 1, 1)
    plt.plot(x_epoch, loss_history, '-o', label='train')
    plt.plot(x_epoch, val_loss_history, '-o', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.xticks(x_epoch)

    plt.subplot(2, 1, 2)
    plt.plot(x_epoch, train_acc_history, '-o', label='train')
    plt.plot(x_epoch, val_acc_history, '-o', label='val')
    plt.plot(x_epoch, [0.5] * len(val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.xticks(x_epoch)

    plt.gcf().set_size_inches(10, 9)
    plt.show()