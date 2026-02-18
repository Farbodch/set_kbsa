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
                    inner as df_inner)
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
                abs as np_abs,
                dot as np_dot,
                ones as np_ones,
                all as np_all,
                any as np_any)
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
    f = df_Function(func_space_V)
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

def indicator_vector_from_function(f, g_constraint):
    """
    Build a distributed dolfin Vector with same layout as u.vector():
      f_indicator_i = 1 if f_i <= g_constraint else 0  (CG1 nodal dofs)
    """
    f_local = f.vector().get_local()
    f_indicator_local = (f_local <= g_constraint).astype(float64)

    f_indicator = df_Vector(f.vector())
    f_indicator.zero()
    f_indicator.set_local(f_indicator_local)
    f_indicator.apply("insert")
    return f_indicator

def integrated_cov_indicator_cg1_mpi_worker(comm,
                                    V, 
                                    mass_matrix, 
                                    u_parent_dir, 
                                    u_A_parent_dir, 
                                    N,
                                    g_constraint,
                                    field_name="temp_field"):
    """
    __summary__

    Args:
        comm (__type__): __discription__
        V (__type__): FunctionSpace(mesh, "CG", 1)
        mass_matrix (__type__): __discription__
        u_parent_dir (__type__): __discription__ REFACTOR!!
        u_A_parent_dir (__type__): __discription__ REFACTOR!!
        N (__type__): __discription__
        g_constraint (__type__): __discription__
        field_name (__type__): __discription__
    Returns:
        spatial_genSobol (float): (1/N) sum(u_indicator_k^T mass_matrix u_A_indicator_k) - mean_u_indicator^T mass_matrix mean_u_A_indicator
    """
    """
    MPI Performance notes:
      - integrated_cov_indicator_cg1_mpi(...) performs N local matvecs and local dot-products.
      - Two allreduces scalars: sum term, mean term.
      - Mean vectors are accumulated as distributed vectors with axpy, no explicit Allreduce needed.
    """
    rank = dolfin_MPI.rank(comm)
    size = dolfin_MPI.size(comm)
    # split sample indices across ranks
    local_indices = np_array_split(np_arange(N, dtype=int), size)[rank]

    # distributed vector accumulators for means
    sum_u_indicator = df_Function(V).vector()
    sum_u_indicator.zero()
    sum_u_A_indicator = df_Function(V).vector()
    sum_u_A_indicator.zero()

    # local scalar accumulator for sum(u_indicator^T mass_matrix u_A_indicator)
    local_sum_inner = 0.0

    u_tmp = df_Function(V)
    u_A_tmp = df_Function(V)

    # ownership range for local dot-products
    # (mass_matrix.mult produces y with same distribution as u_A_indicator)
    # do local dot, then reduce.
    for i in local_indices:
        with XDMFFile(os_path.join(u_parent_dir, f"solution_{i}.xdmf")) as f: #REFACTOR!!!
            f.read_checkpoint(u_tmp, field_name, 0)
        with XDMFFile(os_path.join(u_A_parent_dir, f"solution_{i}.xdmf")) as f: #REFACTOR!!!
            f.read_checkpoint(u_A_tmp, field_name, 0)

        u_indicator = indicator_vector_from_function(u_tmp, g_constraint=g_constraint)
        u_A_indicator = indicator_vector_from_function(u_A_tmp, g_constraint=g_constraint)

        # Accumulate distributed sums for means
        sum_u_indicator.axpy(1.0, u_indicator)
        sum_u_A_indicator.axpy(1.0, u_A_indicator)

        # y = mass_matrix * u_A_indicator
        y = df_Vector(u_A_indicator)
        y.zero()
        mass_matrix.mult(u_A_indicator, y)

        # local dot
        local_sum_inner += float(np_dot(u_indicator.get_local(), y.get_local()))

    # Reduce sum(u_A_indicator^T mass_matrix u_A_indicator)
    global_sum_inner = comm.allreduce(local_sum_inner, op=dolfin_MPI.SUM)

    # Compute means as distributed vectors
    mean_u_indicator = df_Vector(sum_u_indicator)
    mean_u_A_indicator = df_Vector(sum_u_A_indicator)
    mean_u_indicator *= (1.0 / float(N))
    mean_u_A_indicator *= (1.0 / float(N))

    # mean term: mean_u_indicator^T mass_matrix mean_u_A_indicator (do local dot, then reduce)
    y_mean = df_Vector(mean_u_A_indicator)
    y_mean.zero()
    mass_matrix.mult(mean_u_A_indicator, y_mean)
    local_mean_term = float(np_dot(mean_u_indicator.get_local(), y_mean.get_local()))
    global_mean_term = comm.allreduce(local_mean_term, op=dolfin_MPI.SUM)

    spatial_genSobol = (global_sum_inner / float(N)) - global_mean_term
    return float(spatial_genSobol) if rank == 0 else None

def integrated_cov_indicator_cg1_serialized(u_list, u_A_list, mass_matrix, g_constraint):
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
    N = len(u_list)
    A = mass_matrix.array()
    ndofs = A.shape[0]

    sum_inner = 0.0
    sum_u_indicator = np_zeros(ndofs, dtype=float64)
    sum_u_A_indicator = np_zeros(ndofs, dtype=float64)

    for u, ua in zip(u_list, u_A_list):
        u_indicator = (u.vector().get_local() <= g_constraint).astype(float64)
        u_A_indicator = (ua.vector().get_local() <= g_constraint).astype(float64)
        sum_inner += float(u_indicator @ (A @ u_A_indicator))
        sum_u_indicator += u_indicator
        sum_u_A_indicator += u_A_indicator

    mean_u_indicator = sum_u_indicator / float(N)
    mean_u_A_indicator = sum_u_A_indicator / float(N)
    mean_term = float(mean_u_indicator @ (A @ mean_u_A_indicator))
    return (sum_inner / float(N)) - mean_term
