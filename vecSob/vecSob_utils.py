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
from numpy.random import (uniform as np_unif)
from numpy.typing import NDArray
from os import path as os_path
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


def _snap_1d_coords_to_mesh_nodes(coords_1d, lower_bound, upper_bound, mode):
    """
    coords_1d: 1D numpy array of mesh vertex coordinates in a dimension (global, may contain repeats)
    interval_min, interval_max: float bounds, interval_min<=interval_max
    mode: 'lower'|'upper'|'nearest'
       - lower: low_s = smallest node coord >= lower_bound, high_s = largest node coord <= upper_bound
       - upper: low_s = largest  node coord <= lower_bound, high_s = smallest node coord >= upper_bound
       - nearest: low_s = nearest node coord to lower_bound, high_s = nearest node coord to upper_bound

    Returns (low_s, high_s) snapped bounds.
    """
    xs = np_unique(coords_1d)
    xs.sort()

    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    if mode == "lower":
        # Inside-closest: shrink interval to fit within [interval_min, interval_max]
        candidate_min_verts = xs[xs >= lower_bound]
        candidate_max_verts = xs[xs <= upper_bound]
        if len(candidate_min_verts) == 0 or len(candidate_max_verts) == 0:
            # Interval doesn't intersect node coordinates meaningfully; fallback to nearest
            mode = "nearest"
        else:
            low_s = candidate_min_verts[0]
            high_s = candidate_max_verts[-1]
            if low_s > high_s:
                # No node strictly inside both constraints; fallback to nearest
                mode = "nearest"
            else:
                return float(low_s), float(high_s)

    if mode == "upper":
        # Outside-closest: expand interval to cover [lower_bound, upper_bound]
        candidate_min_verts = xs[xs <= lower_bound]
        candidate_max_verts = xs[xs >= upper_bound]
        if len(candidate_min_verts) == 0:
            low_s = xs[0]
        else:
            low_s = candidate_min_verts[-1]
        if len(candidate_max_verts) == 0:
            high_s = xs[-1]
        else:
            high_s = candidate_max_verts[0]
        if low_s > high_s:
            low_s, high_s = high_s, low_s
        return float(low_s), float(high_s)

    # nearest
    low_s = xs[np_argmin(np_abs(xs - lower_bound))]
    high_s = xs[np_argmin(np_abs(xs - upper_bound))]
    if low_s > high_s:
        low_s, high_s = high_s, low_s
    return float(low_s), float(high_s)

def snap_bounds_to_mesh_nodes(mesh, bounds, mode):
    """
    bounds: dict {dim_index: (lo, hi)}; only dims < mesh.geometry().dim() used
    mode: 'lower'|'upper'|'nearest'
    Returns snapped_bounds with same keys for used dims.

    Uses mesh.vertex coordinates (mesh.coordinates()).
    """
    mesh_max_dim = mesh.geometry().dim()
    coords = mesh.coordinates()
    snapped = {}
    for curr_dim, (lower_bound, upper_bound) in bounds.items():
        if curr_dim >= mesh_max_dim:
            continue
        low_s, high_s = _snap_1d_coords_to_mesh_nodes(coords[:, curr_dim], lower_bound, upper_bound, mode)
        snapped[curr_dim] = (low_s, high_s)
    return snapped

def mark_cells_by_midpoint_bounds(mesh, bounds):
    """
    Mark cell = 1 if its midpoint lies within the axis-aligned box defined by bounds.
    bounds: dict {dim_index: (lower_bound, upper_bound)}.
    """
    mesh_max_dim = mesh.geometry().dim()
    cell_markers = MeshFunction("size_t", mesh, mesh_max_dim)
    cell_markers.set_all(0)

    for cell in cells(mesh):
        mid = cell.midpoint().array()
        midpoint_flag = True
        for curr_dim, (lower_bound, upper_bound) in bounds.items():
            if curr_dim < mesh_max_dim and not (lower_bound <= mid[curr_dim] <= upper_bound):
                midpoint_flag = False
                break
        if midpoint_flag:
            cell_markers[cell] = 1
    return cell_markers

def mark_cells_by_nodes_bounds(mesh, bounds, policy='all'):
    """
    Creates a FEniCS CellFunction by marking cells based on whether their
    vertices lie within a specified bounding box.

    This provides a 'node-aligned' region.

    Args:
        mesh (Mesh): The FEniCS mesh object.
        bounds (dict): A dictionary defining the bounding box. Keys are dimension
                       indices (0 for x, 1 for y, 2 for z) and values are tuples (lower_bound, upper_bound).
                       Example: {0: (0.2, 0.8), 1: (0.1, 0.4)}
        policy (str): The rule for including a cell:
                      - 'all': Mark the cell if ALL of its vertices are inside
                               the bounds. This creates an "inner bound" region.
                      - 'any': Mark the cell if AT LEAST ONE of its vertices
                               is inside the bounds. This creates an "outer bound" region.
    Returns:
        MeshFunction: A cell function where cells satisfying the policy are
                      marked with 1, and all others are marked with 0.
    """
    # ensure the policy is valid
    if policy not in ['all', 'any']:
        raise ValueError("Policy must be either 'all' or 'any'.")
    
    # get mesh dimension and geometry data
    max_mesh_dim = mesh.geometry().dim()
    all_vertex_coords = mesh.coordinates()
    cell_to_vertex_map = mesh.cells()
    
    # initialize a MeshFunction to store the integer markers for each cell.
    # the second argument is the topological dimension of the entity to be marked (cells).
    cell_markers = MeshFunction("size_t", mesh, max_mesh_dim)
    # default all cells to marker 0 (outside)
    cell_markers.set_all(0)

    # iterate through each cell by its index, 'cell_idx'
    for cell_idx in range(mesh.num_cells()):
        # get the indices of the vertices that make up this cell
        vertex_indices_for_cell = cell_to_vertex_map[cell_idx]
        # get the actual coordinates of these vertices
        coords_of_cell_vertices = all_vertex_coords[vertex_indices_for_cell]

        #perform the bounds check for all vertices in the cell at once:
        # start by assuming all vertices are inside
        is_inside_flags = np_ones(len(vertex_indices_for_cell), dtype=bool)
        # sequentially apply the filter for each dimension in the bounds
        for curr_dim, (lower_bound, upper_bound) in bounds.items():
            # skip if the dimension is not relevant to the mesh
            if curr_dim >= max_mesh_dim:
                continue
            # get the coordinates for the current dimension for all vertices
            vertex_coords_in_dim = coords_of_cell_vertices[:, curr_dim]     
            # update the flags using a vectorized boolean AND operation.
            # a vertex is only 'still inside' if it was inside before AND it's within the bounds for the current dimension.
            is_inside_flags &= (vertex_coords_in_dim >= lower_bound) & (vertex_coords_in_dim <= upper_bound)
            
        #apply the chosen policy      
        if policy == 'all':
            # np_all() returns True only if every element in the array is True.
            if np_all(is_inside_flags):
                cell_markers[cell_idx] = 1              
        elif policy == 'any':
            # np_any() returns True if at least one element in the array is True.
            if np_any(is_inside_flags):
                cell_markers[cell_idx] = 1
                
    return cell_markers


def assemble_mass_matrix(V, marker=None):
    """
    assemble consistent mass matrix over whole domain (marker None) or subdomain.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = df_Measure("dx", domain=V.mesh(), subdomain_data=marker)
    sub_id = 1 if marker is not None else 0
    mass_matrix = assemble(df_inner(u, v) * dx(sub_id))
    return mass_matrix

def indicator_vector_from_function(u, g_constraint):
    """
    Build a distributed dolfin Vector with same layout as u.vector():
      u_indicator_i = 1 if u_i <= g_constraint else 0  (CG1 nodal dofs)
    """
    u_local = u.vector().get_local()
    u_indicator_local = (u_local <= g_constraint).astype(float64)

    u_indicator = df_Vector(u.vector())
    u_indicator.zero()
    u_indicator.set_local(u_indicator_local)
    u_indicator.apply("insert")
    return u_indicator

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
