from numpy import (ndarray,
                log as np_log, 
                exp as np_exp,
                column_stack as np_column_stack)
from numpy.random import (uniform as np_uniform)

def gen_log_uniform(min_u, max_u, size=None) -> ndarray:
    u = np_uniform(np_log(min_u), np_log(max_u), size=size)
    return np_exp(u)

def gen_uniform(min_u, max_u, size=None) -> ndarray:
    return np_uniform(min_u, max_u, size=size)

def generate_data(distribution_type, min_u, max_u, size=None) -> ndarray:
    if distribution_type == 'uniform':
        return gen_uniform(min_u=min_u, max_u=max_u, size=size)
    if distribution_type == 'log_uniform':
        return gen_log_uniform(min_u=min_u, max_u=max_u, size=size)    
    else:
        raise ValueError(f'Distribution {distribution_type} is not supported.')

def generate_data_multi_dim(domain_specification: list, num_of_samples_n: int) -> ndarray:
    return np_column_stack([generate_data(distribution_type=domain_specification[i]['distribution_type'],
                                        min_u=domain_specification[i]['min'],
                                        max_u=domain_specification[i]['max'], size=num_of_samples_n) 
                            for i in range(len(domain_specification))])