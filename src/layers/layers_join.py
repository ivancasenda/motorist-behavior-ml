from .affine import affine_forward, affine_backward
from .relu import relu_forward, relu_backward
from .normalization import batchnorm_forward, batchnorm_backward
from .dropout import dropout_forward, dropout_backward


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine-batchnorm-relu

    Inputs:
    - x: Input to the affine layer
    - w, b : Weights for the affine layer
    - gamma, beta: Scale and shift for  the batch normalization layer
    - bn_param: Parameter for batch normalization layer

    return:
    - out: Output from the ReLu
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a_norm, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_norm)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache

    da = relu_backward(dout, relu_cache)
    da_norm, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(da_norm, fc_cache)

    return dx, dw, db, dgamma, dbeta


def affine_relu_dropout_forward(x, w, b, dropout_param):
    """
    Convenience layer that performs an affine-relu-dropout
    
    Inputs:
    - x, w, b: Input to the affine layer
    - dropout_param: Parameter for dropout layer

    return:
    - out: Output from dropout
    - cache: Object to give to the backward pass
    """
    z, ar_cache = affine_relu_forward(x, w, b)

    out, dropout_cache = dropout_forward(z, dropout_param)
    cache = (ar_cache, dropout_cache)

    return out, cache


def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu-dropout convenience layer
    """
    ar_cache, dropout_cache = cache
    
    ddropout = dropout_backward(dout, dropout_cache)
    
    return affine_relu_backward(ddropout, ar_cache)


def affine_batchnorm_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    """
    Convenience layer that performs an affine-batchnorm-relu-dropout

    Inputs:
    - x, w, b: Inputs for the affine layer
    - gamma, beta, bn_param: Inputs for the batch normalization layer
    - dropout_param: Parameter for the dropout layer

    return:
    - out: Output from the dropout layer
    - cache: Object to give to the backward pass
    """
    z, abr_cache = affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param)

    out, dropout_cache = dropout_forward(z, dropout_param)
    cache = (abr_cache, dropout_cache)

    return out, cache


def affine_batchnorm_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu-dropout convenience layer
    """
    abr_cache, dropout_cache = cache

    ddropout = dropout_backward(dout, dropout_cache)

    return affine_batchnorm_relu_backward(ddropout, abr_cache)

