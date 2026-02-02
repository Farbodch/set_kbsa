from pathlib import Path
from dolfin import Mesh, FunctionSpace, Function, XDMFFile
from numpy import (int32, float32, float64, uint8, fill_diagonal,
                linspace as np_linspace,
                array as np_array,
                sqrt as np_sqrt,
                asarray as np_asarray,
                einsum as np_einsum,
                stack as np_stack,
                mean as np_mean,
                zeros as np_zeros,
                load as np_load,
                abs as np_abs,
                prod as np_prod,
                where as np_where,
                exp as np_exp,
                sum as np_sum,
                triu_indices as np_triu_indices,
                log as np_log,
                zeros_like as np_zeros_like,
                clip as np_clip,
                all as np_all,
                trace as np_trace,
                packbits as np_packbits)
from numpy.random import (uniform as np_unif)
from numpy.typing import NDArray
from numba.core.registry import CPUDispatcher
from numba import njit, prange
from types import FunctionType
from typing import Union
from time import time_ns
# if needed, import from auxiliary_utils.index_management, DEPRICATE
from utils.other_utils import flipStr, getIndexSuperset, directBinStrSum
from auxiliary_utils.io_management import get_data_file_directories
from auxiliary_utils.mesh_management import get_shortest_geom_mesh_dist
from itertools import product
import gc

def sample_fenics_function(data_directory: str, 
                        mesh: Mesh,
                        func_space_V: FunctionSpace,
                        test_domain: NDArray = np_array([[0,1],[0,0.5]]),
                        g_constraint: float | None = None,
                        num_of_grid_points: int | None = None) -> NDArray:
    """_summary_

    Args:
        data_directory (str): _description_
        mesh (Mesh): _description_
        func_space_V (FunctionSpace): _description_
        test_domain (NDArray, optional): _description_. Defaults to np_array([[0,1],[0,0.5]]).
        g_constraint (float | None, optional): _description_. Defaults to None.
        num_of_grid_points (int | None, optional): _description_. Defaults to None.

    Returns:
        y (NDArray): A numpy array containing the samplings of the target fenics-function at coordinates x_vect.
                     If g_constraint passed-in (not None), then y is a uint8 type binary array (as in, y\in{0,1}^num_of_grid_points).
    """

    test_domain = np_array(test_domain)
    if num_of_grid_points is None:
        h = get_shortest_geom_mesh_dist(mesh=mesh)
        num_of_grid_points = int(2/h)
    sqrt_num_of_grid_points = int(np_sqrt(num_of_grid_points))
    x_vect = np_array(list(product(*[np_linspace(test_domain[i, 0], test_domain[i, 1], sqrt_num_of_grid_points) for i in range(test_domain.shape[0])])))
    field_of_interest = Path(data_directory).name
    f = Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), data_directory) as xdmf_1:
        xdmf_1.read_checkpoint(f, field_of_interest, 0)

    y = np_zeros(num_of_grid_points)
    for i, x in enumerate(x_vect):
        y[i] = f(x[0], x[1])

    del f
    gc.collect()
    
    if g_constraint is not None:
        return (y <= g_constraint).astype(uint8)
    else:
        return y
    
def calculate_vecSob_index_A_einsumed(binary_system_output_data_index_A: NDArray):
    """
    binary_system_output_data_index_A.shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
     # y_I = y_data[0], y_II = y_data[1], y_tilde = y_data[2]. 
    y_data = binary_system_output_data_index_A
    n = y_data.shape[1]
    """
    Next: perform per element averaging.
    y_I_bar = avg(y_data[0,:,:], axis=1),
    y_II_bar = avg(y_data[1,:,:], axis=1),
    y_tilde_bar = avg(y_data[2,:,:], axis=1).
    -> y_X_bar.shape = (h,).
    """
    y_data_bar = y_data.mean(axis=1)

    """
    Next:
        T_A = (1/(n-1))\sum_{i=1}^n\sum_{j=1}^h{(y_I_ij - y_II_bar_j)(y_tilde_ij - y_tilde_bar_j)}
        T = (1/(n-1))\sum_{i=1}^n\sum_{j=1}^h{(y_I_ij-y_I_bar_j)^2}
    """
    y_II_bar = np_asarray(y_data_bar[1], dtype=float64).reshape(1, -1)
    y_tilde_bar = np_asarray(y_data_bar[2], dtype=float64).reshape(1, -1)
    y_I_centered = np_asarray(y_data[0], dtype=float64) - y_II_bar
    y_tilde_centered = np_asarray(y_data[2], dtype=float64) - y_tilde_bar
    T_A = np_einsum('ij,ij->', y_I_centered, y_tilde_centered, optimize=True) / (n - 1)
    
    y_I = np_asarray(y_data[0], dtype=float64)
    y_bar_I = np_asarray(y_data_bar[0], dtype=float64).reshape(1, -1)
    y_I_centered_I = y_I - y_bar_I
    T = np_einsum('ij,ij->', y_I_centered_I, y_I_centered_I, optimize=True) / (n - 1)

    """
    Next:
    S^{vecSob}_A = T_A / T
    """
    return T_A/T


def calculate_vecSob_index_A_vectorized(binary_system_output_data_index_A: NDArray):
    """
    binary_system_output_data_index_A.shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
     # y_I = y_data[0], y_II = y_data[1], y_tilde = y_data[2]. 
    y_data = binary_system_output_data_index_A
    n = y_data.shape[1]
    y_data_bar = y_data.mean(axis=1)

    T_A = ((y_data[0] - y_data_bar[1]) * (y_data[2] - y_data_bar[2])).sum()/(n-1)  
    T = ((y_data[0] - y_data_bar[0])**2).sum()/(n-1)
    return T_A/T

def calculate_vecSob_index_A_looped(binary_system_output_data_index_A: NDArray):
    """
    binary_system_output_data_index_A.shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
     # y_I = y_data[0], y_II = y_data[1], y_tilde = y_data[2]. 
    y_data = binary_system_output_data_index_A
    n = y_data.shape[1]
    h = y_data.shape[2]
    y_data_bar = y_data.mean(axis=1)

    T_A = 0
    T = 0
    for i in range(n):
        for j in range(h):
            T_A += (y_data[0,i,j] - y_data_bar[1,j])*(y_data[2,i,j]-y_data_bar[2,j])
            T += (y_data[0,i,j] - y_data_bar[0,j])**2
    T_A = T_A/(n-1) 
    T = T/(n-1) 
    return T_A/T

def build_sorted_output_data_dict(binary_system_output_data_dict: dict) -> dict:
    """
    Inputs: 
        binary_system_output_data_dict: which has the data for u_I, u_II, 
                                        and the one-hot-encoded indices (e.g., 00001, 00010, ...).
    Returns: 
        sorted_output_data_dict: {one_hot_index_key: NDArray of shape (3, n, num_of_grid_points)} where each
               dict[key].shape = (3, n, h) where [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
    u_I = binary_system_output_data_dict['u_I']
    u_II = binary_system_output_data_dict['u_II']
    sorted_output_data_dict = {}
    for key, arr in binary_system_output_data_dict.items():
        if key in ('u_I', 'u_II'):
            continue
        stacked = np_stack([u_I, u_II, arr], axis=0)   # (3, n, num_of_grid_points)
        sorted_output_data_dict[key] = stacked
    return sorted_output_data_dict