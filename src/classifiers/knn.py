from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange
from math import sqrt


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, vectorized=True):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - vectorized: Whether to use vectorized implementation of compute distance, else uses naive.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if vectorized:
            dists = self.vectorized_compute_distances(X)
        else:
            dists = self.naive_compute_distances(X)

        return self.predict_labels(dists, k=k)

    def naive_compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):

                dists[i, j] = sqrt(np.sum( (X[i] - self.X_train[j]) ** 2 ))

        return dists

    def vectorized_compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # note: L2 (Frobenius) norm of vector x, ||x|| = sqrt(x^2)
        #       If x = a - b, then ||a - b|| = (a - b)^2 = a^2 - 2*a*b + b^2
        #       Therefore, ||A - B|| = A^2 - 2*A*B + B^2
        #       A: (M, D), B: (N, D)
        #       Since np.square does element-wise multiplication, 
        #       matrix^2 has the same dims as matrix
        #       A^2: (M, D), A*B': (N, 1)x(1, M), (B^2)': (1, N)
        
        # L2 distance
        test_square = np.sum(np.square(X), axis = 1).reshape(num_test, 1)
        train_square = np.sum(np.square(self.X_train), axis = 1).reshape(1, num_train)
        
        dists = np.sqrt(test_square + train_square - 2 * np.dot(X, self.X_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            kNearest = np.argsort(dists[i])[:k]
            closest_y = self.y_train[kNearest]
            
            unique, counts = np.unique(closest_y, return_counts=True)
            mostCommonIndex = np.argmax(counts)
            y_pred[i] = unique[mostCommonIndex]

        return y_pred.astype(int)
