from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
from importlib import import_module

import numpy as np

from src.evaluation import metrics


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    train_dset = (tf.data.Dataset) Tensorflow dataset object 
                containing training data and label.
    val_dset = (tf.data.Dataset) Tensorflow dataset object 
                containing validation data and label.
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, train_dset, val_dset,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - scores: Prediction score of each class
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, train_dset, val_dset, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - train_dset: (tf.data.Dataset) Tensorflow dataset object 
                containing training data and label.
        - val_dset: (tf.data.Dataset) Tensorflow dataset object 
                containing validation data and label.

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - num_epochs: The number of epochs to run for during training.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - checkpoint_filepath: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.train_dset = train_dset
        self.val_dset = val_dset

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.lr_decay_every_epoch = kwargs.pop("lr.decay_every_epoch", 1)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.checkpoint_filepath = kwargs.pop("checkpoint_filepath", None)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        optim_package = "src.optimizer"
        optim_module = import_module(f'{optim_package}.{self.update_rule}')
        self.update_rule = getattr(optim_module, self.update_rule)

        self._reset()
        

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self, X_batch, y_batch):
        """
        Make a single gradient update.
        """
        # Compute loss and gradient
        scores, loss, grads = self.model.loss(X_batch, y_batch)
        
        prediction = np.argmax(scores, axis=1)
        accuracy = metrics.accuracy(y_batch, prediction) 

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

        return loss, accuracy


    def _save_checkpoint(self):
        if self.checkpoint_filepath is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "val_loss_history": self.val_loss_history,
            "accuracy_history": self.accuracy_history,
            "val_accuracy_history": self.val_accuracy_history,
        }
        filename = "%s.pkl" % (self.checkpoint_filepath)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)


    def evaluate(self, dset):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        # Compute predictions in batches
        y_true = []
        y_pred = []
        total_loss = 0
        for _,(X_batch, y_batch) in enumerate(dset.as_numpy_iterator()):
            y_true.append(y_batch)
            scores, loss, _= self.model.loss(X_batch, y_batch, training=False)

            total_loss += loss
            y_pred.append(np.argmax(scores, axis=1))
            
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        accuracy = metrics.accuracy(y_true, y_pred)

        return loss, accuracy
        

    def train(self):
        """
        Run optimization to train the model.
        """
        iter_each_epoch = '_'
        for epoch in range(self.num_epochs):
            if self.verbose: print(f'Epoch {epoch+1}/{self.num_epochs}')
            iteration = 0
            train_loss = []
            train_acc = []

            for _, (X_batch, y_batch) in enumerate(self.train_dset.as_numpy_iterator()):
                loss, acc = self._step(X_batch, y_batch)

                train_loss.append(loss)
                train_acc.append(acc)

                # Maybe print training loss
                if self.verbose:
                    template = '\r' + '{}/{} - loss: {:.4f} - accuracy: {:.4f}'
                    print(template.format(iteration, iter_each_epoch, 
                    np.mean(train_loss), 
                    np.mean(train_acc)), 
                    end='')
                iteration += 1 # add iteration

            if epoch == 0: iter_each_epoch = iteration

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            self.epoch += 1
            if self.epoch % self.lr_decay_every_epoch == 0:
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            loss = np.mean(train_loss)
            self.loss_history.append(loss)

            accuracy = np.mean(train_acc)
            self.accuracy_history.append(accuracy)

            #loss, accuracy = self.evaluate(self.train_dset)
            #self.loss_history.append(loss)
            #self.accuracy_history.append(accuracy)

            val_loss, val_accuracy = self.evaluate(self.val_dset)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_accuracy)

            if self.verbose:
                template = '\r' + '{}/{} - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}'
                print(template.format(iteration, iter_each_epoch,
                                    loss,
                                    accuracy,
                                    val_loss,
                                    val_accuracy),
                                    end='\n')

            # Keep track of the best model
            if val_accuracy > self.best_val_acc:
                self._save_checkpoint()
                self.best_val_acc = val_accuracy
                self.best_params = {}
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()


        # At the end of training swap the best params into the model
        self.model.params = self.best_params