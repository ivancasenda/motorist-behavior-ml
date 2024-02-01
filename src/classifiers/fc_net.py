from builtins import range
from builtins import object
import numpy as np

from src.layers.layers_join import affine_relu_forward, affine_relu_backward, affine_batchnorm_relu_forward, affine_batchnorm_relu_backward, \
                                    affine_relu_dropout_forward, affine_relu_dropout_backward, affine_batchnorm_relu_dropout_forward, \
                                    affine_batchnorm_relu_dropout_backward

from src.layers.affine import affine_forward, affine_backward
from src.loss.softmax import softmax_loss


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for curr_layer in range (1, self.num_layers + 1):
            prev_layer = curr_layer - 1
            
            input_layer_dim = layer_dims[prev_layer]
            curr_layer_dim = layer_dims[curr_layer]

            self.params[f'W{curr_layer}'] = weight_scale * np.random.randn(input_layer_dim, curr_layer_dim)
            self.params[f'b{curr_layer}'] = np.zeros(curr_layer_dim)

            if self.normalization == "batchnorm":
                # Batch Norm Parameters
                if curr_layer == self.num_layers: break
                self.params[f'gamma{curr_layer}'] = np.ones(curr_layer_dim)
                self.params[f'beta{curr_layer}'] = np.zeros(curr_layer_dim)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None, training=True):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None or training is False else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        
        caches = []

        curr_input = X
        for curr_layer in range(1, self.num_layers):

            curr_weight = self.params[f'W{curr_layer}']
            curr_bias = self.params[f'b{curr_layer}']

            if self.normalization == "batchnorm":
                curr_gamma = self.params[f'gamma{curr_layer}']
                curr_beta = self.params[f'beta{curr_layer}']
                curr_bn_param = self.bn_params[curr_layer - 1]

                if self.use_dropout:
                    activation_output, curr_cache = affine_batchnorm_relu_dropout_forward(
                        curr_input, curr_weight, curr_bias, curr_gamma, curr_beta, curr_bn_param, self.dropout_param)
                else:
                    activation_output, curr_cache = affine_batchnorm_relu_forward(
                        curr_input, curr_weight, curr_bias, curr_gamma, curr_beta, curr_bn_param)
            else:
                if self.use_dropout:
                    activation_output, curr_cache = affine_relu_dropout_forward(
                        curr_input, curr_weight, curr_bias, self.dropout_param)
                else:
                    activation_output, curr_cache = affine_relu_forward(
                        curr_input, curr_weight, curr_bias)
            
            caches.append(curr_cache)

            curr_input = activation_output

        # Last Layer
        curr_weight = self.params[f'W{self.num_layers}']
        curr_bias = self.params[f'b{self.num_layers}']

        scores, curr_cache = affine_forward(curr_input, curr_weight, curr_bias)
        caches.append(curr_cache)

        # If test mode return early
        if y is None:
            return scores 

        loss, grads = 0.0, {}
        
        # Last Layer (Backward)
        loss, dx_softmax = softmax_loss(scores, y)
        
        loss += 0.5 * self.reg * np.sum(curr_weight * curr_weight)

        if training:
            dCurrInput, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(dx_softmax, caches.pop())
            grads[f'W{self.num_layers}'] += self.reg * curr_weight

        for curr_layer in range(self.num_layers - 1, 0, -1):
            if training:
                if self.normalization == "batchnorm":
                    if self.use_dropout:
                        dCurrInput, grads[f'W{curr_layer}'], grads[f'b{curr_layer}'], grads[f'gamma{curr_layer}'], \
                            grads[f'beta{curr_layer}'] = affine_batchnorm_relu_dropout_backward(dCurrInput, caches.pop())
                    else:
                        dCurrInput, grads[f'W{curr_layer}'], grads[f'b{curr_layer}'], grads[f'gamma{curr_layer}'], \
                            grads[f'beta{curr_layer}'] = affine_batchnorm_relu_backward(dCurrInput, caches.pop())
                else:
                    if self.use_dropout:
                        dCurrInput, grads[f'W{curr_layer}'], grads[f'b{curr_layer}'] = affine_relu_dropout_backward(dCurrInput, caches.pop())
                    else:
                        dCurrInput, grads[f'W{curr_layer}'], grads[f'b{curr_layer}'] = affine_relu_backward(dCurrInput, caches.pop())

            curr_weight = self.params[f'W{curr_layer}']

            loss += 0.5 * self.reg * np.sum(curr_weight * curr_weight)

            if training: grads[f'W{curr_layer}'] += self.reg * curr_weight

        return scores, loss, grads

    
    def predict(self, X):
        scores = self.loss(X)

        return scores