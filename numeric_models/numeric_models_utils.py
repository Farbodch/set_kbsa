from numpy import (log as np_log, exp as np_exp)
from numpy.random import (uniform as np_uniform)

def gen_log_uniform(min_u, max_u, size=None):
    u = np_uniform(np_log(min_u), np_log(max_u), size=size)
    return np_exp(u)

def gen_uniform(min_u, max_u, size=None):
    return np_uniform(min_u, max_u, size=size)

def generate_data(distribution_type, min_u, max_u, size=None):
    if distribution_type == 'uniform':
        return gen_uniform(min_u=min_u, max_u=max_u, size=size)
    if distribution_type == 'log_uniform':
        return gen_log_uniform(min_u=min_u, max_u=max_u, size=size)    
    else:
        raise ValueError(f'Distribution {distribution_type} is not supported.')