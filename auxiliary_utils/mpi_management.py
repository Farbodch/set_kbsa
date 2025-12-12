


def get_padding_for_hsic(n: int=1, 
                n_max: int=1, 
                size: int=1, 
                verbose: bool=True):
    
    n_padded_up = n + (size-n%size)
    n_padded_down = n - (n%size)

    if n_padded_up > n_max:
        if verbose:
            print(f'Using {n-n_padded_down} less data points to ensure MPI parallelization. Need {n_padded_up} data points at the next parallelization increment using {size} MPI tasks/core.')
        n = n_padded_down
    else:
        if verbose:
            print(f'Using {n_padded_up-n} extra data points to ensure MPI parallelization.')
        n = n_padded_up
    return n

def get_per_rank_padded_indices(N: int=1, 
                            size: int=1, 
                            rank: int=0) -> tuple[list, int]:
    
    max_jobs = (N + size - 1) // size 
    indices = range(rank, N, size)
    padded_indices = [i for i in indices]
    num_of_padding = 0

    while len(padded_indices) < max_jobs:
        padded_indices.append(None)
        num_of_padding += 1

    return padded_indices, num_of_padding

def get_total_num_of_padding(N: int=1, 
                            size: int=1) -> int:
    
    max_jobs = (N + size - 1) // size
    total_num_of_padding = 0
    for r in range(size):
        indices = range(r, N, size)
        if len(indices) < max_jobs:
            total_num_of_padding += 1
    return total_num_of_padding