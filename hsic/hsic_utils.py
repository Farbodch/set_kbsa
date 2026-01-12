from pathlib import Path
from dolfin import Mesh, FunctionSpace, Function, XDMFFile
from numpy import (int32, float32, uint8, fill_diagonal,
                array as np_array,
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
from utils.other_utils import flipStr, getIndexSuperset, directBinStrSum
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
    f = Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), data_directory) as xdmf_1:
        xdmf_1.read_checkpoint(f, field_of_interest, 0)

    f_samplings = np_zeros(m)
    for i, x in enumerate(x_vect):
        f_samplings[i] = f(x[0], x[1])

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
    return [flipStr(i) for i in sorted(getIndexSuperset(dim_of_U, higher_order=higher_order), key=lambda x: directBinStrSum(x))]
