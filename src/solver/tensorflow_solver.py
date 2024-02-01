import tensorflow as tf 
from builtins import object
import datetime


class TensorflowSolver(object):
    """ Performs training for tensorflow model """
    def __init__(self, model, optimizer, train_dset, val_dset, **kwargs):
        """ 
        Construct a new TensorflowSolver instance.
        
        Inputs:
        - model: A tensorflow model object
        - optimizer: Tensorflow optimizer object
        - train_dset: (tf.data.Dataset) Tensorflow dataset object 
                containing training data and label.
        - val_dset: (tf.data.Dataset) Tensorflow dataset object 
                containing validation data and label.
        
        Optional inputs:
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - use_tensorboard:
        - logs_directory:
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dset = train_dset
        self.val_dset = val_dset

        # Unpack keyword arguments
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.use_tensorboard = kwargs.pop("use_tensorboard", False)
        self.logs_directory = kwargs.pop("logs_directory", "tensorboard/logs/gradient_tape/") 
        self.checkpoint_directory = kwargs.pop("checkpoint_filepath", "models/checkpoints/")

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()
        

    def _reset(self):
        model_name = type(self.model).__name__
        self.filepath = self.checkpoint_directory + model_name
        self.best_val_acc = 0
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    
    def _save_model(self):
        self.model.save(self.filepath)
        if self.verbose:
            print('Saving best model checkpoint to "%s"' % self.filepath)


    def train(self, device):
        """
        Simple training loop for use with models defined using tf.keras. It trains
        a model for one epoch on the CIFAR-10 training set and periodically checks
        accuracy on the CIFAR-10 validation set.
        
        Inputs:
        - model_init_fn: A function that takes no parameters; when called it
        constructs the model we want to train: model = model_init_fn()
        - optimizer_init_fn: A function which takes no parameters; when called it
        constructs the Optimizer object we will use to optimize the model:
        optimizer = optimizer_init_fn()
        - num_epochs: The number of epochs to train for
        
        Returns: Nothing, but prints progress during training
        """
        with tf.device(device):
            
            if self.use_tensorboard:
                # Setup tensorboard log file
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                train_log_dir = self.logs_directory + current_time + '/train'
                val_log_dir = self.logs_directory + current_time + '/val'
                train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                val_summary_writer = tf.summary.create_file_writer(val_log_dir)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
            val_loss = tf.keras.metrics.Mean(name='val_loss')
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
            
            iter_each_epoch = '_'
            
            for epoch in range(self.num_epochs):
                
                # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
                train_loss.reset_states()
                train_accuracy.reset_states()
                val_loss.reset_states()
                val_accuracy.reset_states()
                
                if self.verbose: print(f'Epoch {epoch+1}/{self.num_epochs}')
                t = 0

                for x_np, y_np in self.train_dset:
                    with tf.GradientTape() as tape:
                        
                        # Use the model function to build the forward pass.
                        scores = self.model(x_np, training=True)
                        t_loss = loss_fn(y_np, scores)
        
                        gradients = tape.gradient(t_loss, self.model.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                        
                        # Update the metrics
                        train_loss.update_state(t_loss)
                        train_accuracy.update_state(y_np, scores)
                        
                        if self.verbose:
                            template = '\r' + '{}/{} - loss: {:.4f} - accuracy: {:.4f}'
                            print(template.format(t, iter_each_epoch, train_loss.result(), train_accuracy.result()), end='')

                        t += 1

                if epoch == 0: iter_each_epoch = t
                
                for val_x, val_y in self.val_dset:
                    # During validation at end of epoch, training set to False
                    prediction = self.model(val_x, training=False)
                    v_loss = loss_fn(val_y, prediction)

                    val_loss.update_state(v_loss)
                    val_accuracy.update_state(val_y, prediction)
                
                loss_result = train_loss.result()
                accuracy_result = train_accuracy.result()
                val_loss_result = val_loss.result()
                val_accuracy_result = val_accuracy.result()

                # Append history
                self.loss_history.append(loss_result)
                self.train_acc_history.append(accuracy_result)

                self.val_loss_history.append(val_loss_result)
                self.val_acc_history.append(val_accuracy_result)

                if self.use_tensorboard:
                    # Write for tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('epoch_accuracy', accuracy_result, step=epoch)
                        tf.summary.scalar('epoch_loss', loss_result, step=epoch)

                    with val_summary_writer.as_default():
                        tf.summary.scalar('epoch_accuracy', val_accuracy_result, step=epoch)
                        tf.summary.scalar('epoch_loss', val_loss_result, step=epoch)

                if self.verbose:
                    template = '\r' + '{}/{} - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}'
                    print(template.format(t, iter_each_epoch,
                                        loss_result,
                                        accuracy_result,
                                        val_loss_result,
                                        val_accuracy_result),
                                        end='\n')
                
                if val_accuracy_result > self.best_val_acc:
                    self.best_val_acc = val_accuracy_result
                    self._save_model()