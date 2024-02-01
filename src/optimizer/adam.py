import numpy as np 


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    m = config['m']
    v = config['v']
    t = config['t'] + 1

    m = beta1 * m + (1-beta1) * dw
    mt = m / (1-beta1 ** t)
    v = beta2 * v + (1-beta2) * (dw ** 2)
    vt = v / (1 - beta2 ** t)
    next_w = w - learning_rate * mt / (np.sqrt(vt) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_w, config