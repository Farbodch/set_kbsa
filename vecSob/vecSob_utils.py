from pathlib import Path
from dolfin import (Mesh, 
                    MeshFunction, 
                    cells,
                    FunctionSpace, 
                    TestFunction, 
                    TrialFunction, 
                    assemble,
                    XDMFFile,
                    MPI as dolfin_MPI,
                    Function as df_Function,
                    Vector as df_Vector,
                    Measure as df_Measure,
                    inner as df_inner,
                    Point as df_Point)
from numpy import (int32, float32, float64, uint8, fill_diagonal,
                linspace as np_linspace,
                array as np_array,
                sqrt as np_sqrt,
                asarray as np_asarray,
                einsum as np_einsum,
                stack as np_stack,
                unique as np_unique,
                argmin as np_argmin,
                arange as np_arange,
                array_split as np_array_split,
                zeros as np_zeros,
                zeros_like as np_zeros_like,
                abs as np_abs,
                dot as np_dot,
                ones as np_ones,
                all as np_all,
                any as np_any,
                min as np_min,
                max as np_max,
                True_ as np_true,
                empty as np_empty)
from dolfin.cpp.mesh import MeshFunctionSizet
from numpy.random import (uniform as np_unif)
from numpy.typing import NDArray
from os import path as os_path
from numba.core.registry import CPUDispatcher
from numba import njit, prange
from types import FunctionType
from typing import Union
from time import time_ns
from auxiliary_utils.io_management import get_data_file_directories
from auxiliary_utils.mesh_management import get_shortest_geom_mesh_dist
from itertools import product
from mpi4py import MPI as pyMPI
import gc

def check_array_values_in_interval_1d(x_vect: NDArray, 
                                   test_domain: NDArray,
                                   return_which_failed: bool = False) -> bool:
    lower_bound = np_min(test_domain)
    upper_bound = np_max(test_domain)
    x_vect = x_vect.flatten()
    in_interval_mask = (x_vect >= lower_bound) & (x_vect <= upper_bound)
    all_values_in = np_all(in_interval_mask)
    outlier_values = x_vect[~in_interval_mask]
    if return_which_failed:
        return all_values_in, outlier_values
    else:
        return all_values_in

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
    # sqrt_num_of_grid_points = int(np_sqrt(num_of_grid_points))
    x_vect = np_array(list(product(*[np_linspace(test_domain[i, 0], test_domain[i, 1], num_of_grid_points) for i in range(test_domain.shape[0])])))
    all_in_bool = check_array_values_in_interval_1d(x_vect=x_vect, test_domain=test_domain)
    if all_in_bool != np_true:
        raise ValueError(f'x_vect generated values out of the interval {test_domain}')
    field_of_interest = Path(data_directory).name
    f = df_Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), data_directory) as xdmf_1:
        xdmf_1.read_checkpoint(f, field_of_interest, 0)
    y = np_zeros(num_of_grid_points)
    for i, x in enumerate(x_vect):
        eval_point = df_Point(*x)
        y[i] = f(eval_point)
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
        T_A = (1/(n-1))\sum_{i=1}^n\sum_{j=1}^h{(y_II_ij - y_II_bar_j)(y_tilde_ij - y_tilde_bar_j)}
        T = (1/(n-1))\sum_{i=1}^n\sum_{j=1}^h{(y_I_ij-y_I_bar_j)^2}
    """
    y_II_bar = np_asarray(y_data_bar[1], dtype=float64).reshape(1, -1)
    y_tilde_bar = np_asarray(y_data_bar[2], dtype=float64).reshape(1, -1)
    y_II_centered = np_asarray(y_data[1], dtype=float64) - y_II_bar
    y_tilde_centered = np_asarray(y_data[2], dtype=float64) - y_tilde_bar
    T_A = np_einsum('ij,ij->', y_II_centered, y_tilde_centered, optimize=True)
    
    y_I = np_asarray(y_data[0], dtype=float64)
    y_bar_I = np_asarray(y_data_bar[0], dtype=float64).reshape(1, -1)
    y_I_centered_I = y_I - y_bar_I
    T = np_einsum('ij,ij->', y_I_centered_I, y_I_centered_I, optimize=True)

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

    T_A = ((y_data[1] - y_data_bar[1]) * (y_data[2] - y_data_bar[2])).sum()
    T = ((y_data[0] - y_data_bar[0])**2).sum()
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
            T_A += (y_data[1,i,j] - y_data_bar[1,j])*(y_data[2,i,j]-y_data_bar[2,j])
            T += (y_data[0,i,j] - y_data_bar[0,j])**2
    T_A = T_A
    T = T
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

#------------------------------------------------------------------------------------------
# Functions relevant to the integral formulation of spatio-general aggregated sobol indices
#------------------------------------------------------------------------------------------
def load_fenics_functions_as_indicator(comm, 
                                     V: FunctionSpace, 
                                     data_dirs_to_eval_list: list,
                                     g_constraint: float):
    rank = comm.Get_rank()
    N = len(data_dirs_to_eval_list)
    size = pyMPI.INT.Get_size()
    # rank 0 hosts memory for counter, others allocate 0 bytes
    window_MPI = pyMPI.Win.Allocate(size if rank == 0 else 0, comm=comm)
    if rank == 0:
        window_MPI.Lock(rank)
        window_MPI.Put(np_zeros(1, dtype='i'), rank)
        window_MPI.Unlock(rank)
    comm.Barrier()

    my_idx = np_empty(1, dtype='i')
    increment = np_ones(1, dtype='i')

    local_data = [] # each entry stores (index, u_array, uA_array)
    g_tmp = df_Function(V)

    while True:
        window_MPI.Lock(0)
        window_MPI.Fetch_and_op(increment, my_idx, 0, op=pyMPI.SUM)
        window_MPI.Unlock(0)
        i = my_idx[0]
        if i >= N:
            break
        data_directory_i = data_dirs_to_eval_list[i]
        field_of_interest = Path(data_directory_i).name
        with XDMFFile(pyMPI.COMM_SELF, data_dirs_to_eval_list[i]) as f:
            f.read_checkpoint(g_tmp, field_of_interest, 0)
        g_indicator = (g_tmp.vector().get_local() <= g_constraint).astype(float64)
        local_data.append((i, g_indicator))

    window_MPI.Free()
    allgathered_data = comm.allgather(local_data)
    flattened = [item for sublist in allgathered_data for item in sublist]
    flattened.sort(key=lambda x: x[0])
    indicator_fens_list = [x[1] for x in flattened]

    return indicator_fens_list

def integrated_cov_indicator_cg1_serialized(f_list, f_A_list, mass_matrix, g_constraint):
    """
    __summary__
    Args:
        u_list (list): __discription__
        u_A_list (list): __discription__
        mass_matrix (__type__): __discription__
        g_constraint (float): __discription__
    Return:
      spatial_genSobol (float): (1/N) sum(u_indicator^T mass_matrix u_A_indicator) - mean_u_indicator^T mass_matrix mean_u_A_indicator 
    """
    """
    Performance note:
        - Uses mass_matrix.array() (dense). For testing purposes ONLY where ndofs is small.
    """
    N = len(f_list)
    A = mass_matrix.array()
    ndofs = A.shape[0]

    sum_inner = 0.0
    sum_u_indicator = np_zeros(ndofs, dtype=float64)
    sum_u_A_indicator = np_zeros(ndofs, dtype=float64)

    for u, ua in zip(f_list, f_A_list):
        u_indicator = (u.vector().get_local() <= g_constraint).astype(float64)
        u_A_indicator = (ua.vector().get_local() <= g_constraint).astype(float64)
        sum_inner += float(u_indicator @ (A @ u_A_indicator))
        sum_u_indicator += u_indicator
        sum_u_A_indicator += u_A_indicator

    mean_u_indicator = sum_u_indicator / float(N)
    mean_u_A_indicator = sum_u_A_indicator / float(N)
    mean_term = float(mean_u_indicator @ (A @ mean_u_A_indicator))
    return (sum_inner / float(N)) - mean_term

def integrated_spatial_cov_memory_mpi(comm, 
                                        mass_matrix, 
                                        g_indicators_list: list, 
                                        g_A_indicators_list: list):
    """Computes spatial_cov = (1/N) sum(I_g^T M I_{g_A}) - mean_I_g^T M mean_I_{g_A}"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = len(g_indicators_list)

    d_vec, y_vec = mass_matrix.createVecs()
    d_vec_I_g = d_vec.duplicate()

    ndofs = d_vec.getSize()
    
    local_sum_inner = 0.0
    local_sum_I_g = np_zeros(ndofs, dtype=float64)
    local_sum_I_g_A = np_zeros(ndofs, dtype=float64)

    for i in range(rank, N, size):
        I_g = g_indicators_list[i]
        I_g_A = g_A_indicators_list[i]
        
        # assign I_g_A array to PETSc, perform mass_matrix * uA_ind -> y_vec
        d_vec.setArray(I_g_A)
        mass_matrix.mult(d_vec, y_vec)
        
        # assign I_g array to PETSc, perform I_g^T * y_vec -> inner_prod
        d_vec_I_g.setArray(I_g)
        inner_prod = d_vec_I_g.dot(y_vec)
        
        local_sum_inner += inner_prod
        
        # accumulate sums for the means
        local_sum_I_g += I_g
        local_sum_I_g_A += I_g_A

    global_sum_inner = comm.allreduce(local_sum_inner, op=pyMPI.SUM)
    
    global_sum_I_g = np_zeros_like(local_sum_I_g)
    global_sum_I_g_A = np_zeros_like(local_sum_I_g_A)
    
    comm.Allreduce(local_sum_I_g, global_sum_I_g, op=pyMPI.SUM)
    comm.Allreduce(local_sum_I_g_A, global_sum_I_g_A, op=pyMPI.SUM)

    mean_I_g = global_sum_I_g / float(N)
    mean_I_g_A = global_sum_I_g_A / float(N)

    d_vec.setArray(mean_I_g_A)
    mass_matrix.mult(d_vec, y_vec)
    d_vec_I_g.setArray(mean_I_g)
    global_mean_term = d_vec_I_g.dot(y_vec)
    
    spatial_cov = (global_sum_inner / float(N)) - global_mean_term

    return spatial_cov if rank == 0 else None

#------------------------------------------------------------------------------------------
# deprecated fens below
#------------------------------------------------------------------------------------------
# def indicator_vector_from_function(f, g_constraint):
#     """
#     Build a distributed dolfin Vector with same layout as f.vector():
#       f_indicator_i = 1 if f_i <= g_constraint else 0  (CG1 nodal dofs)
#     """
#     f_local = f.vector().get_local()
#     f_indicator_local = (f_local <= g_constraint).astype(float64)

#     f_indicator = df_Vector(f.vector())
#     f_indicator.zero()
#     f_indicator.set_local(f_indicator_local)
#     f_indicator.apply("insert")
#     return f_indicator
# def integrated_cov_indicator_cg1_mpi_worker_dynamic(comm, 
#                                                     V, 
#                                                     mass_matrix, 
#                                                     f_parent_dir, 
#                                                     f_A_parent_dir, 
#                                                     N, 
#                                                     g_constraint, 
#                                                     field_name="temp_field"):
#     rank = comm.Get_rank()
#     size = pyMPI.INT.Get_size()
#     # Rank 0 hosts memory for counter, others allocate 0 bytes
#     mpi_window = pyMPI.Win.Allocate(size if rank == 0 else 0, comm=comm)
#     if rank == 0:
#         mpi_window.Lock(rank)
#         mpi_window.Put(np_zeros(1, dtype='i'), rank)
#         mpi_window.Unlock(rank)
#     comm.Barrier()

#     sum_f_indicator = df_Function(V).vector()
#     sum_f_indicator.zero()
#     sum_f_A_indicator = df_Function(V).vector()
#     sum_f_A_indicator.zero()

#     local_sum_inner = 0.0
#     f_tmp = df_Function(V)
#     f_A_tmp = df_Function(V)

#     my_idx = np_empty(1, dtype='i')
#     increment = np_ones(1, dtype='i')

#     # dynamic work loop
#     while True:
#         mpi_window.Lock(0)
#         mpi_window.Fetch_and_add(increment, my_idx, 0)
#         mpi_window.Unlock(0)

#         i = my_idx[0]

#         if i >= N:
#             break
        
#         with XDMFFile(os_path.join(f_parent_dir, f"solution_{i}.xdmf")) as f: 
#             f.read_checkpoint(f_tmp, field_name, 0)
#         with XDMFFile(os_path.join(f_A_parent_dir, f"solution_{i}.xdmf")) as f: 
#             f.read_checkpoint(f_A_tmp, field_name, 0)
        
#         f_indicator = indicator_vector_from_function(f_tmp, g_constraint=g_constraint)
#         f_A_indicator = indicator_vector_from_function(f_A_tmp, g_constraint=g_constraint)

#         sum_f_indicator.axpy(1.0, f_indicator)
#         sum_f_A_indicator.axpy(1.0, f_A_indicator)
        
#         y = df_Vector(f_A_indicator)
#         y.zero()
#         mass_matrix.mult(f_A_indicator, y)
#         local_sum_inner += f_indicator.inner(y)
#     #//end while loop
#     mpi_window.Free()

#     global_sum_inner = comm.allreduce(local_sum_inner, op=pyMPI.SUM)
#     local_sum_f_arr = sum_f_indicator.get_local()
#     local_sum_f_A_arr = sum_f_A_indicator.get_local()
    
#     global_sum_f_arr = np_zeros_like(local_sum_f_arr)
#     global_sum_f_A_arr = np_zeros_like(local_sum_f_A_arr)
    
#     comm.Allreduce(local_sum_f_arr, global_sum_f_arr, op=pyMPI.SUM)
#     comm.Allreduce(local_sum_f_A_arr, global_sum_f_A_arr, op=pyMPI.SUM)
    
#     mean_f_indicator = df_Vector(sum_f_indicator)
#     mean_f_indicator.set_local(global_sum_f_arr / float(N))
#     mean_f_indicator.apply("insert")
    
#     mean_f_A_indicator = df_Vector(sum_f_A_indicator)
#     mean_f_A_indicator.set_local(global_sum_f_A_arr / float(N))
#     mean_f_A_indicator.apply("insert")
    
#     y_mean = df_Vector(mean_f_A_indicator)
#     y_mean.zero()
#     mass_matrix.mult(mean_f_A_indicator, y_mean)
    
#     global_mean_term = mean_f_indicator.inner(y_mean)
    
#     spatial_genSobol = (global_sum_inner / float(N)) - global_mean_term
#     return float(spatial_genSobol) if rank == 0 else None


# def integrated_cov_indicator_cg1_mpi_worker(comm,
#                                     V, 
#                                     mass_matrix, 
#                                     u_parent_dir, 
#                                     u_A_parent_dir, 
#                                     N,
#                                     g_constraint,
#                                     field_name="temp_field"):
#     """
#     __summary__

#     Args:
#         comm (__type__): __discription__
#         V (__type__): FunctionSpace(mesh, "CG", 1)
#         mass_matrix (__type__): __discription__
#         u_parent_dir (__type__): __discription__ REFACTOR!!
#         u_A_parent_dir (__type__): __discription__ REFACTOR!!
#         N (__type__): __discription__
#         g_constraint (__type__): __discription__
#         field_name (__type__): __discription__
#     Returns:
#         spatial_genSobol (float): (1/N) sum(u_indicator_k^T mass_matrix u_A_indicator_k) - mean_u_indicator^T mass_matrix mean_u_A_indicator
#     """
#     """
#     MPI Performance notes:
#       - integrated_cov_indicator_cg1_mpi(...) performs N local matvecs and local dot-products.
#       - Two allreduces scalars: sum term, mean term.
#       - Mean vectors are accumulated as distributed vectors with axpy, no explicit Allreduce needed.
#     """
#     rank = dolfin_MPI.rank(comm)
#     size = dolfin_MPI.size(comm)
#     # split sample indices across ranks
#     local_indices = np_array_split(np_arange(N, dtype=int), size)[rank]

#     # distributed vector accumulators for means
#     sum_u_indicator = df_Function(V).vector()
#     sum_u_indicator.zero()
#     sum_u_A_indicator = df_Function(V).vector()
#     sum_u_A_indicator.zero()

#     # local scalar accumulator for sum(u_indicator^T mass_matrix u_A_indicator)
#     local_sum_inner = 0.0

#     u_tmp = df_Function(V)
#     u_A_tmp = df_Function(V)

#     # ownership range for local dot-products
#     # (mass_matrix.mult produces y with same distribution as u_A_indicator)
#     # do local dot, then reduce.
#     for i in local_indices:
#         with XDMFFile(os_path.join(u_parent_dir, f"solution_{i}.xdmf")) as f: #REFACTOR!!!
#             f.read_checkpoint(u_tmp, field_name, 0)
#         with XDMFFile(os_path.join(u_A_parent_dir, f"solution_{i}.xdmf")) as f: #REFACTOR!!!
#             f.read_checkpoint(u_A_tmp, field_name, 0)

#         u_indicator = indicator_vector_from_function(u_tmp, g_constraint=g_constraint)
#         u_A_indicator = indicator_vector_from_function(u_A_tmp, g_constraint=g_constraint)

#         # Accumulate distributed sums for means
#         sum_u_indicator.axpy(1.0, u_indicator)
#         sum_u_A_indicator.axpy(1.0, u_A_indicator)

#         # y = mass_matrix * u_A_indicator
#         y = df_Vector(u_A_indicator)
#         y.zero()
#         mass_matrix.mult(u_A_indicator, y)

#         # local dot
#         local_sum_inner += float(np_dot(u_indicator.get_local(), y.get_local()))

#     # Reduce sum(u_A_indicator^T mass_matrix u_A_indicator)
#     global_sum_inner = comm.allreduce(local_sum_inner, op=dolfin_MPI.SUM)

#     # Compute means as distributed vectors
#     mean_u_indicator = df_Vector(sum_u_indicator)
#     mean_u_A_indicator = df_Vector(sum_u_A_indicator)
#     mean_u_indicator *= (1.0 / float(N))
#     mean_u_A_indicator *= (1.0 / float(N))

#     # mean term: mean_u_indicator^T mass_matrix mean_u_A_indicator (do local dot, then reduce)
#     y_mean = df_Vector(mean_u_A_indicator)
#     y_mean.zero()
#     mass_matrix.mult(mean_u_A_indicator, y_mean)
#     local_mean_term = float(np_dot(mean_u_indicator.get_local(), y_mean.get_local()))
#     global_mean_term = comm.allreduce(local_mean_term, op=dolfin_MPI.SUM)

#     spatial_genSobol = (global_sum_inner / float(N)) - global_mean_term
#     return float(spatial_genSobol) if rank == 0 else None