import time

def time_function(fn, *args):
    """
    Measure the time it took to execute a function fn with args.
    
    Inputs:
    - fn: Function to be measured
    - *args: Function parameter to be passed.

    Returns:
    - measured_time: time it took to execute the function.
    """
    tic = time.time()
    fn(*args)
    toc = time.time()
    return toc - tic