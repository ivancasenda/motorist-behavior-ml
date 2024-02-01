from builtins import range
import numpy as np


def _normalization_forward(x, gamma, beta, eps):
    #mean = np.sum(x, axis=0) / N
    mean = np.mean(x, axis=0)

    x_minus_mean = x - mean
    
    #var = np.sum(np.square(x_minus_mean), axis=0) / N
    var = np.var(x, axis=0)
    
    std = np.sqrt(var + eps)
    
    inv_std = 1.0 / std

    normalized = (x_minus_mean) * inv_std

    scale_shift_normalized = gamma * normalized + beta

    out = scale_shift_normalized
    cache = (normalized, gamma, x_minus_mean, mean, var, inv_std, std, x)

    return out, cache


def _normalization_backward(dout, cache, axis):
    normalized, gamma, x_minus_mean, _, _, inv_std, _, _ = cache

    N, D = dout.shape

    dbeta = np.sum(dout, axis=axis) 

    dscale = dout   
    dgamma = np.sum(normalized * dscale, axis=axis)

    dnormalized = gamma * dscale # (N, D)
    
    dinv_std = np.sum(x_minus_mean * dnormalized, axis=0) # (D, )
    dstd = -dinv_std * (inv_std ** 2)
    dsqrt_var = 0.5 * inv_std * dstd
    dvar = 1.0/N * np.ones((N,D)) * dsqrt_var
    dsquare = 2 * (x_minus_mean) * dvar

    dminus_mean1 = inv_std * dnormalized
    dminus_mean2 = dsquare
    dminus_mean = dminus_mean1 + dminus_mean2

    dx1 = dminus_mean

    dmean = -dminus_mean
    dx2 = np.ones((N,D)) * (np.sum(dmean, axis=0) / N)

    dx = dx1 + dx2

    return dx, dgamma, dbeta


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    _, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":

        out, cache = _normalization_forward(x, gamma, beta, eps)
        _, _, _, mean, var, _, _, _ = cache

        running_mean = momentum * running_mean + (1-momentum) * mean
        running_var = momentum * running_var + (1-momentum) * var

    elif mode == "test":

        normalized = (x - running_mean) / np.sqrt(running_var + eps)
        scale_shift_normalized = gamma * normalized + beta

        out = scale_shift_normalized

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    column_axis = 0
    dx, dgamma, dbeta = _normalization_backward(dout, cache, axis=column_axis)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    
    normalized, gamma, x_minus_mean, _, _, _, std, _ = cache
    N, _ = dout.shape

    dbeta = np.sum(dout, axis=0)
    
    dscale = dout
    dgamma = np.sum(normalized * dscale, axis=0)

    dmean = 1/N * np.sum(dscale, axis=0)
    dvar = 2/N * np.sum(x_minus_mean * dscale, axis=0)
    dstd = dvar / (2 * std)
    dx = gamma * ((dscale - dmean) * std - dstd * (x_minus_mean)) / (std ** 2)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)

    out, cache = _normalization_forward(x.T, gamma.reshape(-1,1), beta.reshape(-1,1), eps)
    out = out.T

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    row_axis = 1
    dx, dgamma, dbeta = _normalization_backward(dout.T, cache, row_axis)
    dx = dx.T

    return dx, dgamma, dbeta