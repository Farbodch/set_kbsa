from pathlib import Path
from dolfin import (Mesh, 
                    FunctionSpace,
                    XDMFFile,
                    UserExpression as df_UserExpression,
                    inner as df_inner,
                    Measure as df_Measure,
                    assemble as df_assemble,
                    Constant as df_Constant,
                    Function as df_Function,
                    TrialFunction as df_TrialFunction,
                    TestFunction as df_TestFunction,
                    as_backend_type as df_as_backend_type,
                    Point as df_Point)
from dolfin.cpp.mesh import MeshFunctionSizet
from numpy import (int32, float64, float32, uint8, fill_diagonal,
                array as np_array,
                mean as np_mean,
                zeros as np_zeros,
                save as np_save,
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
                packbits as np_packbits,
                empty as np_empty,
                logical_xor as np_logical_xor,
                bitwise_xor as np_bitwise_xor,
                dot as np_dot,
                concatenate as np_concatenate,
                array_split as np_array_split)
from numpy.random import (uniform as np_unif)
from numpy.typing import NDArray
from numba.core.registry import CPUDispatcher
from numba import njit, prange
from types import FunctionType
from typing import Union
from time import time_ns
from shutil import rmtree
from mpi4py import MPI
from auxiliary_utils.index_management import flip_str, get_index_superset, direct_binstr_sum
from auxiliary_utils.io_management import get_data_file_directories
import gc

def get_mesh_bounds(mesh):
    mesh_coords = mesh.coordinates()
    xmin, ymin = mesh_coords.min(axis=0)
    xmax, ymax = mesh_coords.max(axis=0)
    return (xmin, xmax), (ymin, ymax)
def sample_fenics_function(data_directory: str, 
                        mesh: Mesh,
                        func_space_V: FunctionSpace,
                        test_domain: NDArray = np_array([[0,1],[0,0.5]]), 
                        num_of_spatial_sampling_m: int = 5,
                        g_constraint: float = None,
                        rng = np_unif):
    m = num_of_spatial_sampling_m
    test_domain = np_array(test_domain)
    
    x_vect = rng(test_domain[:, 0], test_domain[:, 1], size=(m, test_domain.shape[0]))
    field_of_interest = Path(data_directory).name
    f = df_Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), data_directory) as xdmf_1:
        xdmf_1.read_checkpoint(f, field_of_interest, 0)

    f_samplings = np_zeros(m)
    for i, x in enumerate(x_vect):
        eval_point = df_Point(*x)
        # f_samplings[i] = f(x[0], x[1])
        f_samplings[i] = f(eval_point)

    del f
    gc.collect()
    
    if g_constraint is not None:
        return (f_samplings <= g_constraint).astype(uint8)
    else:
        return f_samplings
def sample_analytical_function(u: NDArray,  
                            process_generator: Union[FunctionType, CPUDispatcher] = None,
                            test_domain: NDArray = np_array([[0,1]]), 
                            num_of_spatial_sampling_m: int = 5,
                            g_constraint: float = None,
                            x_vect: NDArray = None,
                            rng=np_unif):
    m = num_of_spatial_sampling_m
    test_domain = np_array(test_domain)
    if x_vect is None:
        x_vect = rng(low=test_domain[:, 0], high=test_domain[:, 1], size=(m, test_domain.shape[0]))
    # f_samplings = np_zeros(m)
    f = process_generator(u)
    f_samplings = f(x_vect)
    if len(f_samplings.shape) == 2:
        f_samplings = f_samplings.squeeze(-1)
    #FAULTY LOGIC! This will copy result into np.zeros of size m!!
    # f_samplings[:] = result #too redundant? Maybe remove.
    if g_constraint is not None:
        return (f_samplings <= g_constraint).astype(uint8)
    else:
        return f_samplings
def get_process_model_output_data_analytical(u_arr: NDArray, 
                                        process_generator: Union[FunctionType, CPUDispatcher] = None,
                                        test_domain: NDArray = np_array([[0,1]]),
                                        num_of_spatial_sampling_m: int = 5,
                                        g_constraint: float = None,
                                        fix_x_vect: bool = False,
                                        rng=np_unif) -> NDArray:
    
    n = u_arr.shape[0]
    m = num_of_spatial_sampling_m
    x_vect = None
    if fix_x_vect:
        x_vect = rng(low=test_domain[:, 0], high=test_domain[:, 1], size=(m, test_domain.shape[0]))
    system_output_data = np_zeros((n, num_of_spatial_sampling_m), dtype=uint8)
    for i, u_i in enumerate(u_arr):
        system_output_data[i] = sample_analytical_function(u=u_i,
                                                        process_generator=process_generator,
                                                        test_domain=test_domain,
                                                        num_of_spatial_sampling_m=m,
                                                        g_constraint=g_constraint,
                                                        x_vect=x_vect,
                                                        rng=rng)
    return system_output_data   
def _get_direct_sums(data_array: NDArray, axis=1) -> NDArray:
    """input: nxm ndarray -> output: row sums /w output shape (n,)"""
    return data_array.sum(axis=axis) 
@njit(parallel=True)
def _get_gram_matrix(binary_data_array: NDArray, 
                    popcount_8bit=None) -> NDArray:
    """
    input: nxm ndarray -> output: pairwise AND count /w output shape (n,n)
    effectly, performing binary_data_array @ binary_data_array.T    
    """
    n, m = binary_data_array.shape
    gram_matrix = np_zeros((n, n), dtype=int32)
    if popcount_8bit is not None:
        for i in prange(n):
            for j in range(i, n):
                s = 0
                for k in range(m):
                    s += popcount_8bit[binary_data_array[i, k] & binary_data_array[j, k]]
                gram_matrix[i, j] = s
                gram_matrix[j, i] = s
    else:
        for i in prange(n):
            for j in range(n):
                s = 0
                for k in range(m):
                    s += binary_data_array[i, k] & binary_data_array[j, k]
                gram_matrix[i, j] = s
    return gram_matrix
    #-----
    # return binary_data_array @ binary_data_array.T
def _get_XOR_count(binary_direct_sums: NDArray, pairwise_AND_count: NDArray) -> NDArray:
    """xOr count using identity AxOrB=a+b-2(aANDb)"""
    xOr_count = binary_direct_sums[:, None] + binary_direct_sums[None, :] - 2*pairwise_AND_count
    return xOr_count.astype(float32)
def approximate_set_lebesgue(binary_system_output_data: NDArray,
                            lambda_X: float,
                            verbose: bool=True,
                            with_popcount: bool=True) -> NDArray:  
    _, m = binary_system_output_data.shape
    #row sums /w output shape (n,)
    if verbose:
            t0_0 = time_ns()
    binary_direct_sums = _get_direct_sums(data_array=binary_system_output_data)
    if verbose:
            t0_1 = time_ns()
    #pairwise AND count /w output shape (n,n)
    if with_popcount:
        popcount_8bit = np_array([bin(i).count("1") for i in range(256)], dtype=uint8)
        pairwise_and_count = _get_gram_matrix(binary_data_array=np_packbits(binary_system_output_data, axis=1), 
                                              popcount_8bit=popcount_8bit)
    else:
        pairwise_and_count = _get_gram_matrix(binary_data_array=binary_system_output_data)
    if verbose:
            t1_1 = time_ns()
    #xOr count using identity AxOrB=a+b-2(aANDb)
    xOr_count = _get_XOR_count(binary_direct_sums=binary_direct_sums,
                               pairwise_AND_count=pairwise_and_count)
    if verbose:
            t2_1 = time_ns()
            dt__get_direct_sums = (t0_1-t0_0) / 1e9
            # print(f'Wall-clock time of _get_direct_sums: {dt__get_direct_sums:.3f} (s)')
            dt__get_gram_matrix = (t1_1-t0_1) / 1e9
            print(f'Wall-clock time of dt__get_gram_matrix: {dt__get_gram_matrix:.3f} (s)')
            dt__get_XOR_count = (t2_1-t1_1) / 1e9
            # print(f'Wall-clock time of dt__get_XOR_count: {dt__get_XOR_count:.3f} (s)')
    return lambda_X*(xOr_count/m)

class IndicatorDifferenceSquared(df_UserExpression):
        def __init__(self, f_1, f_2, g_constraint, **kwargs):
            super().__init__(**kwargs)
            self.f_1 = f_1
            self.f_2 = f_2
            self.g_constraint = g_constraint
        def eval(self, values, x):
            is_in_S = 1.0 if self.f_1(x) <= self.g_constraint else 0.0
            is_in_T = 1.0 if self.f_2(x) <= self.g_constraint else 0.0
            values[0] = (is_in_S - is_in_T)**2
        def value_shape(self):
            return ()

def calculate_symmetric_difference_serialized(f_1: df_Function, 
                                          f_2: df_Function, 
                                          g_constraint: float, 
                                          V: FunctionSpace, 
                                          marker=None) -> float:
    """
    Calculates ∫(indicator_S - indicator_T)² dx using quadrature integration.
    
    Args:
        f_1 (df_Function): First solution.
        f_2 (df_Function): Second solution.
        g_constraint (float): The inequality constraint defining the excursion set.
        V (FunctionSpace): The function space.
        marker (__type__, optional): __discription__

    Returns:
        float: The scalar value of the integral.
    """
    mesh = V.mesh()
    #isolate to subset(sub-interval) of interest according to passed-in mesh markers.
    dx_sub = df_Measure('dx', domain=mesh, subdomain_data=marker)
    integral_id = 1 if marker is not None else 0
    integrand = IndicatorDifferenceSquared(f_1=f_1, f_2=f_2, g_constraint=g_constraint, degree=1)
    integral_value = df_assemble(integrand * dx_sub(integral_id))
    return integral_value

def compute_nodal_weights(V: FunctionSpace, marker: MeshFunctionSizet = None):
    mesh = V.mesh()
    dx = df_Measure('dx', domain=mesh, subdomain_data=marker)
    integral_id = 1 if marker is not None else 0
    weights_vec = df_assemble(df_Constant(1.0) * df_TestFunction(V) * dx(integral_id))
    return weights_vec.get_local()

def load_fenics_functions_as_indicator(comm, 
                                       V: FunctionSpace, 
                                       data_dirs_to_eval_list: list, 
                                       g_constraint: float):
    data_dirs_to_eval_list.sort()
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = len(data_dirs_to_eval_list)
    local_indices = np_array_split(range(N), size)[rank]
    local_data = []
    my_fenics_fen = df_Function(V)


    for i in local_indices:
        data_directory_i = data_dirs_to_eval_list[i]
        field_of_interest = Path(data_directory_i).name
        with XDMFFile(MPI.COMM_SELF, data_directory_i) as f:
            f.read_checkpoint(my_fenics_fen, field_of_interest, 0)
        indicator_fen = (my_fenics_fen.vector().get_local() <= g_constraint).astype(uint8)
        local_data.append(indicator_fen)
    allgathered_data = comm.allgather(local_data) 
    indicator_fens_list = [item for sublist in allgathered_data for item in sublist]
    #Returning a list of N NDArrays
    return indicator_fens_list

def compute_integrated_gamma_matrix_mpi(comm,
                                    mass_matrix,
                                    indicator_fens_list: list,
                                    chunk_size: int,
                                    tmp_checkpoint_dir: str,
                                    restart: bool = False):
    """
    Calculates all pairwise symmetric difference integrals ∫(S(x)_i - T(x)_j)² dx
    in a parallel, vectorized, and chunked manner.
    
    Args:
        comm: MPI communicator.
        V (df.FunctionSpace): The function space, contains both basis element and mesh information.
        mass_matrix (PETSc.Mat): The consistent mass matrix of our FEM mesh elements, 
                                 over the whole domain or subset of domain, depending on passed-in matrix.
        indicator_fens_list: (list): List of NDArray, containing the coefficients of the FE expansion of the indicator functions.
        chunk_size (int): How many 'j' vectors to load into memory at once.

    Returns:
        On rank 0: A numpy NDArray of shape (N, N) containing the symmetric matrix of all pairwise symmetric difference integrals.
        On others: An empty numpy NDArray.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = len(indicator_fens_list)
    d_vec, y_vec = mass_matrix.createVecs()
    
    curr_rank_tmp_dir = Path(tmp_checkpoint_dir) / f"rank_{rank}"
    curr_rank_tmp_dir.mkdir(parents=True, exist_ok=True)

    local_i_indices_list = []
    local_j_indices_list = []
    local_results = []
    #outer loop -> distributed statically between MPI workers.
    for i in range(rank, N, size):
        checkpoint_file = curr_rank_tmp_dir / f"row_{i:05d}.npy"
        if restart and checkpoint_file.exists():
            #if the file exists and this is an instance of the function called as a 'restart', 
            #then this row for this data point has already been processed. Skip.
            continue

        checkpoint_integral_row_i = np_zeros(N, dtype=float64)
        f_indicator_i = indicator_fens_list[i]
        #inner loop (only upper triangle of j > i computed) -> chunked
        for j_start in range(i + 1, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            for j in range(j_start, j_end):
                f_indicator_j = indicator_fens_list[j]
                #XOR on int8 data -> float64 for PETSc
                symm_diff = np_bitwise_xor(f_indicator_i, f_indicator_j).astype(float64)

                #assign from numpy array to PETSc
                d_vec.setArray(symm_diff)

                #perform global matrix-vector product (neighbour comms & halo exchange concerns) .
                #this performs the mass_matrix*(S_i - T_j) and assigns to y_vec

                mass_matrix.mult(d_vec, y_vec)
                #perform global dot-product. a global sum reduction is also perform.
                #this performs (S_i - T_j)^T*y_vec, which based on previous line of code,
                #we are effectively calculating (S_i - T_j)^T*mass_matrix*(S_i - T_j)
                symm_diff_integral = d_vec.dot(y_vec)

                local_i_indices_list.append(i)
                local_j_indices_list.append(j)
                local_results.append(symm_diff_integral)
                checkpoint_integral_row_i[j] = symm_diff_integral
        np_save(checkpoint_file, checkpoint_integral_row_i)
    local_i_indices_arr = np_array(local_i_indices_list, dtype=int32)
    local_j_indices_arr = np_array(local_j_indices_list, dtype=int32)
    local_results_arr = np_array(local_results, dtype=float64)
    gathered_i_indices_arr = comm.gather(local_i_indices_arr, root=0)
    gathered_j_indices_arr = comm.gather(local_j_indices_arr, root=0)
    gathered_results_arr = comm.gather(local_results_arr, root=0)

    if rank == 0 and not restart:
        gamma_matrix = np_zeros((N, N), dtype=float64)

        all_i_indices = np_concatenate(gathered_i_indices_arr)
        all_j_indices = np_concatenate(gathered_j_indices_arr)
        all_results = np_concatenate(gathered_results_arr)
    
        assert len(all_i_indices) == len(all_j_indices)
        assert len(all_i_indices) == len(all_results)

        #fill upper-triangle of matrix
        gamma_matrix[all_i_indices, all_j_indices] = all_results
        #fill lower-triangle of matrix, enforce symmetry
        gamma_matrix[all_j_indices, all_i_indices] = all_results

        return gamma_matrix
    else:
        return np_empty((0, 0))

def cleanup_checkpoint_files(comm, 
                             tmp_checkpoint_dir: str):
    rank = comm.Get_rank()
    if rank != 0:
        return
    tmp_checkpoint_dir = Path(tmp_checkpoint_dir)

    if not tmp_checkpoint_dir.exists():
        print(f"Cleanup skipped. Directory '{tmp_checkpoint_dir}' does not exist.")
        return
    try:
        rmtree(tmp_checkpoint_dir)
        print(f"Successfully cleaned-up (deleted) checkpoint directory {tmp_checkpoint_dir}.")
    except OSError as e:
        print(f"Error occured in deleting directory {tmp_checkpoint_dir}. Reason: {e}")


def assemble_gamma_matrix_from_checkpoints(comm, 
                                           tmp_checkpoint_dir: str, 
                                           n: int,
                                           share_with_all_MPI_ranks: bool = False):
    rank = comm.Get_rank()
    if not share_with_all_MPI_ranks and rank != 0:
        return None
    
    tmp_checkpoint_dir = Path(tmp_checkpoint_dir)
    gamma_matrix = np_zeros((n, n), dtype=float64)
    checkpoints_paths_list_found_flag = False

    if rank == 0:    
        checkpoints_paths_list = list(tmp_checkpoint_dir.rglob("row_*.npy"))
        if len(checkpoints_paths_list) > 0:
            checkpoints_paths_list_found_flag = True
    
    if share_with_all_MPI_ranks:
        checkpoints_paths_list_found_flag = comm.bcast(checkpoints_paths_list_found_flag, root=0)
    
    if not checkpoints_paths_list_found_flag:
        raise ValueError("Warning: no checkpoint files found!")

    if rank == 0:
        print(f"Found {len(checkpoints_paths_list)} checkpoint files to assemble gamma_matrix with.")
        for cp_path in checkpoints_paths_list:
            try:
                row_idx = int(cp_path.stem.split('_')[1])
                row_data = np_load(cp_path)
                #fill upper-triangle of matrix
                gamma_matrix[row_idx, :] += row_data
                #fill lower-triangle of matrix, enforce symmetry
                gamma_matrix[:, row_idx] += row_data
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse file {cp_path}. Error: {e}")
                continue
    
    if share_with_all_MPI_ranks:
        comm.Bcast(gamma_matrix, root=0)
    
    return gamma_matrix

def load_data_and_calculate_symmetric_difference_mpi_worker(comm, 
                                        V, 
                                        mass_matrix, 
                                        data_dirs_to_eval_list: list,  
                                        g_constraint: float, 
                                        chunk_size: int) -> NDArray:
    """
    Calculates all pairwise symmetric difference integrals ∫(S(x)_i - T(x)_j)² dx
    in a parallel, vectorized, and chunked manner.
    
    Args:
        comm: MPI communicator.
        V (df.FunctionSpace): The function space, contains both basis element and mesh information.
        mass_matrix (PETSc.Mat): The consistent mass matrix of our FEM mesh elements, 
                                 over the whole domain or subset of domain, depending on passed-in matrix.
        data_dirs_to_eval_list: (list): List of strings, containing the directory to the N solution files.
        g_constraint (float): The inequality constraint defining the excursion set.
        chunk_size (int): How many 'j' vectors to load into memory at once.

    Returns:
        On rank 0: A numpy NDArray of shape (N, N) containing the symmetric matrix of all pairwise symmetric difference integrals.
        On others: An empty numpy NDArray.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    d_vec, y_vec = mass_matrix.createVecs() 
    local_i_indices_list = []
    local_j_indices_list = []
    local_results = []

    f_i = df_Function(V)
    f_j = df_Function(V)

    N = len(data_dirs_to_eval_list)
    #outer loop -> distributed statically between MPI workers.
    for i in range(rank, N, size):
        data_directory_i = data_dirs_to_eval_list[i]
        field_of_interest = Path(data_directory_i).name
        with XDMFFile(data_directory_i) as f:
            f.read_checkpoint(f_i, field_of_interest, 0)
        f_indicator_i = (f_i.vector().get_local() <= g_constraint)
        #inner loop -> chunked
        for j_start in range(i + 1, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            for j in range(j_start, j_end):
                data_directory_j = data_dirs_to_eval_list[j]
                with XDMFFile(data_directory_j) as f:
                    f.read_checkpoint(f_j, field_of_interest, 0)
                f_indicator_j = (f_j.vector().get_local() <= g_constraint)

                #XOR to get difference
                diff_local = np_logical_xor(f_indicator_i, f_indicator_j).astype(float64)
                #assign from numpy array to PETSc
                d_vec.setArray(diff_local)
                
                #perform global matrix-vector product (neighbour comms & halo exchange concerns) .
                #this performs the mass_matrix*(S_i - T_j) and assigns to y_vec
                mass_matrix.mult(d_vec, y_vec)

                #perform global dot-product. a global sum reduction is also perform.
                #this performs (S_i - T_j)^T*y_vec, which based on previous line of code,
                #we are effectively calculating (S_i - T_j)^T*mass_matrix*(S_i - T_j)
                global_integral = d_vec.dot(y_vec)
                
                local_i_indices_list.append(i)
                local_j_indices_list.append(j)
                local_results.append(global_integral)

    local_i_indices_arr = np_array(local_i_indices_list, dtype=int32)
    local_j_indices_arr = np_array(local_j_indices_list, dtype=int32)
    local_results_arr = np_array(local_results, dtype=float64)
    gathered_i_indices_arr = comm.gather(local_i_indices_arr, root=0)
    gathered_j_indices_arr = comm.gather(local_j_indices_arr, root=0)
    gathered_results_arr = comm.gather(local_results_arr, root=0)

    if rank == 0:
        all_i_indices = np_concatenate(gathered_i_indices_arr)
        all_j_indices = np_concatenate(gathered_j_indices_arr)
        all_results = np_concatenate(gathered_results_arr)
        
        assert len(all_i_indices) == len(all_j_indices)
        assert len(all_i_indices) == len(all_results)

        gamma_matrix = np_zeros((N, N), dtype=float64)
        #fill upper-triangle of matrix
        gamma_matrix[all_i_indices, all_j_indices] = all_results
        #fill lower-triangle of matrix, enforce symmetry
        gamma_matrix[all_j_indices, all_i_indices] = all_results

        return gamma_matrix
    else:
        return np_empty((0, 0))

def get_K_gamma(process_type: str = 'fenics_function',
                data_dirs_to_eval_list: list = None,
                n: int = None,
                num_of_spatial_sampling_m: int = None,
                mesh: Mesh = None,
                func_space_V: FunctionSpace = None,
                test_domain: NDArray = np_array([[0,1],[0,0.5]]),
                g_constraint: float = None,
                verbose: bool = False,
                binary_system_output_data = None,
                process_generator: Union[FunctionType, CPUDispatcher] = None,
                u_arr: NDArray = None,
                with_popcount: bool = True):
    if binary_system_output_data is None:
        m = num_of_spatial_sampling_m
        if verbose:
            t0_0 = time_ns()
        if process_type == 'fenics_function':
            for i, dir in enumerate(data_dirs_to_eval_list):
                binary_system_output_data = np_zeros((n, m), dtype=uint8)
                binary_system_output_data[i] = sample_fenics_function(data_directory=dir,
                                                                    mesh=mesh,
                                                                    func_space_V=func_space_V,
                                                                    test_domain=test_domain,
                                                                    num_of_spatial_sampling_m=m,
                                                                    g_constraint=g_constraint)
        if process_type == 'analytical':
            try:
                assert u_arr is not None
            except AssertionError:
                print('u_arr was NOT passed-in function K_Gamma, but is required when process_type is \'analytical\'.')
                raise AssertionError
            binary_system_output_data = get_process_model_output_data_analytical(u_arr=u_arr,
                                                                            process_generator=process_generator,
                                                                            test_domain=test_domain,
                                                                            num_of_spatial_sampling_m=m,
                                                                            g_constraint=g_constraint,
                                                                            rng=np_unif)
            # for i, u_i in enumerate(u_arr):
            #     binary_system_output_data[i] = sample_analytical_function(u=u_i,
            #                                                             process_generator=process_generator,
            #                                                             test_domain=test_domain,
            #                                                             num_of_spatial_sampling_m=m,
            #                                                             g_constraint=g_constraint)
        if verbose: t0_1 = time_ns()
    if g_constraint:
        try: 
            assert np_all((binary_system_output_data==0) | (binary_system_output_data==1)), "Non-binary values detected"
        except AssertionError as e:
            print('An error occured during sampling of process-model in \'binary_system_output_data\'. Expected only binary outputs (comparison with g_constraint), but non-binary values were returned.')
            raise
    lambda_X = 1
    for i in range(len(test_domain)):
        lambda_curr_dom = max(test_domain[i])-min(test_domain[i])
        if lambda_curr_dom != 0:
            lambda_X *= lambda_curr_dom
    if verbose: t1_0 = time_ns()
    lambda_matrix = approximate_set_lebesgue(binary_system_output_data=binary_system_output_data, 
                                                lambda_X=lambda_X,
                                                with_popcount=with_popcount)
    if verbose: t1_1 = time_ns()
    sigma_squared = np_mean(lambda_matrix)
    if verbose: t2_1 = time_ns()
    K_gamma = np_exp(-1.0*lambda_matrix/(2*sigma_squared))
    if verbose: t3_1 = time_ns()
    fill_diagonal(K_gamma, 0.0)
    if verbose:
        t4_1 = time_ns()
        if binary_system_output_data is None:
            dt_spatial_sampling = (t0_1-t0_0) / 1e9
            print(f'Wall-clock time of spatial_sampling: {dt_spatial_sampling:.3f} (s)')
        dt_approximate_set_lebesgue = (t1_1-t1_0) / 1e9
        dt_sigma_squared = (t2_1-t1_1) / 1e9
        dt_exponentiate_K_gamma = (t3_1-t2_1) / 1e9
        dt_fill_diag_of_K_gamma = (t4_1-t3_1) / 1e9
        print(f'Wall-clock time of approximate_set_lebesgue: {dt_approximate_set_lebesgue:.3f} (s)')
        print(f'Wall-clock time of sigma_squared: {dt_sigma_squared:.3f} (s)')
        print(f'Wall-clock time of exponentiate_K_gamma: {dt_exponentiate_K_gamma:.3f} (s)')
        print(f'Wall-clock time of fill_diag_of_K_gamma: {dt_fill_diag_of_K_gamma:.3f} (s)')
    return K_gamma
def sobolev_kernel_univar(u_i, u_j):
    du = u_i-u_j
    return (1.0 + (u_i-0.5)*(u_j-0.5) + 0.5*(du*du-np_abs(du)+1.0/6.0))
def get_K_U_sobolev_looped(data_directory: str | None = None,
                    n: int = 5,
                    num_of_inputs: int = 5,
                    which_input_one_hot: str = '00001',
                    input_data=None,
                    verbose: bool = False):
    if verbose: t0 = time_ns()    
    dim_U = num_of_inputs
    if input_data is None:
        input_data_dirs_list = get_data_file_directories(data_directory, data_type='input_data')
        u_arr = np_zeros((n, dim_U))
        for i, dir in enumerate(input_data_dirs_list):
            u_arr[i] = np_load(dir)
    else:
        u_arr = input_data
    which_input_mask = np_array([c == '1' for c in which_input_one_hot], dtype=bool)
    active_dims = np_where(which_input_mask)[0]
    if len(active_dims) == 0:
        raise ValueError("which_input_one_hot must include at least one dimension.")
    K_sob_matrix = np_zeros((n,n), dtype=float)
    for j in range(n):
        for i in range(j+1):
            # K_ij = 1.0
            counter = 0
            for d in active_dims:
                u_i = u_arr[i, d]
                u_j = u_arr[j, d]
                if counter == 0:
                    K_ij = sobolev_kernel_univar(u_i, u_j)
                else:
                    K_ij *= sobolev_kernel_univar(u_i, u_j)
                counter = 1
            K_sob_matrix[i,j] = K_ij-1.0
            K_sob_matrix[j,i] = K_ij-1.0
    fill_diagonal(K_sob_matrix, 0.0)
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of get_K_U_sobolev_looped: {dt:.3f} (s)')
    return K_sob_matrix
def get_K_U_sobolev_vectorized(data_directory: str | None = None,
                    n: int = 5,
                    num_of_inputs: int = 5,
                    which_input_one_hot: str = '00001',
                    chunk_size: int | None = None,
                    input_data: NDArray | None = None,
                    verbose: bool = False) -> NDArray:
    if verbose: t0 = time_ns()
    dim_U = num_of_inputs
    if input_data is None:
        assert data_directory is not None, "Must input a parent data directory if no explicit input_data is not passed in."
        input_data_dirs_list = get_data_file_directories(data_directory, data_type='input_data')
        u_arr = np_zeros((n, dim_U))
        for i, dir in enumerate(input_data_dirs_list):
            u_arr[i] = np_load(dir)
    else:
        u_arr = input_data
    which_input_mask = np_array([c == '1' for c in which_input_one_hot], dtype=bool)
    u_active = u_arr[:, which_input_mask]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = n
    K_sob_matrix = np_zeros((n,n), dtype=float)
    start_idx = 0
    while start_idx < n:
        end_idx = min(start_idx + chunk_size, n) 
        u_i = u_active[start_idx:end_idx, None, :]
        u_j = u_active[None, :, :]
        du = u_i - u_j
        K_curr_chunk_sobolev = (1.0 + (u_i-0.5)*(u_j-0.5) + 0.5*(du*du-np_abs(du)+1.0/6.0))
        K_sob_matrix[start_idx:end_idx, :] = np_prod(K_curr_chunk_sobolev, axis=2)-1.0
        start_idx = end_idx
    fill_diagonal(K_sob_matrix, 0.0)
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of get_K_U_sobolev_vectorized with chunks of {chunk_size}: {dt:.3f} (s)')
    return K_sob_matrix
def calculate_hsic_looped(K_U: NDArray, K_gamma: NDArray, verbose: bool = False) -> float:
    if verbose:
        t0 = time_ns()
    n = K_U.shape[0]
    HSIC_A = 0.0
    const_0 = 2.0 / (n * (n - 1))

    for j in range(1, n):
        for i in range(j):
            HSIC_A += K_U[i, j] * K_gamma[i, j]
    HSIC_A = const_0*HSIC_A
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of calculate_hsic_looped: {dt:.3f} (s)')
    return HSIC_A
def calculate_hsic_vectorized(K_U: NDArray, K_gamma: NDArray, verbose: bool = False) -> float:
    if verbose:
        t0 = time_ns()
    n = K_U.shape[0]
    const_0 = 2.0 / (n * (n - 1))
    #upper-triangle indices (i < j)
    up_tri_indices_i, up_tri_indices_j = np_triu_indices(n, k=1)
    HSIC_A = K_U[up_tri_indices_i, up_tri_indices_j] * K_gamma[up_tri_indices_i, up_tri_indices_j]
    HSIC_A = const_0 * np_sum(HSIC_A)
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of calculate_hsic_vectorized: {dt:.3f} (s)')
    return HSIC_A
def calculate_hsic_traced(K_U: NDArray, K_gamma: NDArray, verbose: bool = False) -> float:
    if verbose:
        t0 = time_ns()
    n = K_U.shape[0]
    const_0 = (1/(n*(n-1)))
    HSIC_A = const_0*np_trace(K_U @ K_gamma)
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of calculate_hsic_method_2: {dt:.3f} (s)')
    return HSIC_A
def transform_logUnif_to_unitUnif(min_u, max_u, log_unif_samples):
    transformed_samples = (np_log(log_unif_samples) - np_log(min_u))/(np_log(max_u)-np_log(min_u))
    transformed_samples = np_clip(transformed_samples, 0.0, 1.0)
    eps = 1e-12
    try:
        assert np_sum((transformed_samples>1+eps)) + np_sum((transformed_samples<0-eps))==0
    except AssertionError:
        print("logUnif_to_unitUnif Assertion FAILED!")
        print("Min transformed:", transformed_samples.min())
        print("Max transformed:", transformed_samples.max())
        print("Values < 0:", transformed_samples[transformed_samples < 0])
        print("Values > 1:", transformed_samples[transformed_samples > 1])
        raise
    return transformed_samples
def transform_unif_to_unitUnif(min_u, max_u, unif_samples):
    transformed_samples = (unif_samples-min_u)/(max_u-min_u)
    transformed_samples = np_clip(transformed_samples, 0.0, 1.0)
    eps = 1e-12
    try:
        assert np_sum((transformed_samples>1+eps)) + np_sum((transformed_samples<0-eps))==0
    except AssertionError:
        print("unif_to_unitUnif Assertion FAILED!")
        print("Min transformed:", transformed_samples.min())
        print("Max transformed:", transformed_samples.max())
        print("Values < 0:", transformed_samples[transformed_samples < 0])
        print("Values > 1:", transformed_samples[transformed_samples > 1])
        raise
    return transformed_samples
def transform_all_u_inputs(u_arr: NDArray, u_domain_specifications: list):
    n, d = u_arr.shape
    assert d == len(u_domain_specifications)

    u_arr_transformed = np_zeros_like(u_arr, dtype=float)

    for u_idx in range(d):
        u_spec = u_domain_specifications[u_idx]
        u_curr_idx = u_arr[:, u_idx]
        min_u = u_spec['min']
        max_u = u_spec['max']
        try:
            assert min_u < max_u
        except AssertionError:
            print(f"Invalid min and max for distribution of input u_{u_idx}. min_u {min_u} is NOT less than max_u {max_u}!")
            raise
        if u_spec['distribution_type'] == 'log_uniform':
            try:
                assert min_u > 0
            except AssertionError:
                print(f"Invalid min for log_uniform distribution of input u_{u_idx}. min_u must be larger than is, {min_u} is NOT!")
                raise
            u_arr_transformed[:, u_idx] = transform_logUnif_to_unitUnif(min_u, max_u, u_curr_idx)
        elif u_spec['distribution_type'] == 'uniform':
            u_arr_transformed[:, u_idx] = transform_unif_to_unitUnif(min_u, max_u, u_curr_idx)
        else:
            raise ValueError(f"Unknown distribution type '{u_spec['distribution_type']}' for dimension {u_idx}.") 
    return u_arr_transformed
def get_u_index_superset_one_hot_binstrs(dim_of_U, higher_order=False):
    return [flip_str(i) for i in sorted(get_index_superset(dim_of_U, higher_order=higher_order), key=lambda x: direct_binstr_sum(x))]
