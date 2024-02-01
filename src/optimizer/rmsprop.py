import numpy as np 


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None

    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']
    epsilon = config['epsilon']
    cache = config['cache']

    cache = decay_rate * cache + (1-decay_rate) * (dw ** 2)
    next_w = w - learning_rate * dw / (np.sqrt(cache) + epsilon)

    config['cache'] = cache

    return next_w, config