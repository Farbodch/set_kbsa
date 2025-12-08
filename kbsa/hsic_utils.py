from pathlib import Path
import ast
from dolfin import Mesh, FunctionSpace, Function, XDMFFile
from numpy import ndarray, float32, uint8, fill_diagonal, array as np_array
from numpy import mean as np_mean
from numpy import zeros as np_zeros
from numpy import load as np_load
from numpy.random import uniform as np_unif
from numpy import abs as np_abs
from numpy import prod as np_prod
from numpy import where as np_where
from numpy import exp as np_exp
from numpy import sum as np_sum
from numpy import triu_indices as np_triu_indices
from numpy import log as np_log
from numpy import zeros_like as np_zeros_like
from time import time_ns
from utils.other_utils import flipStr, getIndexSuperset, directBinStrSum
import gc

def parse_meta_data(meta_file: Path):
    """
    Extract and return cdr_params dict from meta_data.txt.
    """
    text = meta_file.read_text()
    #cdr_params appear after "cdr_params_"
    start_idx = text.find("cdr_params_")
    if start_idx == -1:
        return None
    start_idx += len("cdr_params_")
    #extract param dictionary content between '{' and '}'
    dict_str = text[start_idx:].strip()
    dict_str = dict_str[dict_str.find("{"): dict_str.rfind("}")+1]

    try:
        params = ast.literal_eval(dict_str)
        return params
    except:
        print(f"Could not parse cdr_params in {meta_file}")
        return None

def get_data_file_dirs(base_dir: str, data_type: str):
    """
    data_type ∈ {'fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data'}
    Returns a list of file paths for chosen_cdr_field across all qualified sub_sub_directories.
    """
    base_dir = Path(base_dir)   

    if data_type not in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data']:
        raise ValueError("chosen cdr field must be one of 'fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data'")
    

    target_filename = data_type
    if data_type in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']:
        target_filename += '.h5'
    else:
        target_filename += '.npy'
    # target_filename = f"{chosen_cdr_field}.h5"

    collected_paths = []

    num_of_parent_skips = 0
    num_of_sub_folder_skips = 0
    num_of_sub_sub_folder_skips = 0
    #loop over each parent directory inside base_dir
    for parent in base_dir.iterdir():
        if not parent.is_dir():
            num_of_parent_skips += 1
            continue

        #1) check parent/meta_data.txt
        meta_file = parent / "meta_data.txt"
        if not meta_file.exists():
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: no meta_data.txt")
            continue

        params = parse_meta_data(meta_file)
        if params is None:
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: could not parse params.")
            continue

        #validate required conditions
        if not (
            params.get("t_end") == 0.05 and
            params.get("num_steps") == 500 and
            params.get("return_bool") is False
        ):
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: incorrect cdr_params")
            continue

        #2) check subdirectories
        sub_dirs = [d for d in parent.iterdir() if d.is_dir()]

        #skip if parent folder has no subfolder structure
        if len(sub_dirs) == 0:
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: no sub-directories found.")
            continue

        for sub in sub_dirs:
            sub_subs = [d for d in sub.iterdir() if d.is_dir()]
            if len(sub_subs) != 6:
                num_of_sub_folder_skips += 1
                # print(f"Skipping {parent}/{sub}: does not have 6 sub_sub-folders.")
                continue

            #3)check inside each sub_sub_directory for exactly 10 files
            for sub_sub in sub_subs:
                files = [f for f in sub_sub.iterdir() if f.is_file()]
                if len(files) != 10:
                    num_of_sub_sub_folder_skips += 1
                    # print(f"Skipping {parent}/{sub}/{sub_sub}: does not contain 9 files.")
                    continue

                #4)collect chosen .h5 or .npy file
                wanted_path = sub_sub / target_filename
                if wanted_path.exists():
                    if data_type == 'input_data':
                        collected_paths.append(str(wanted_path))
                    else:
                        collected_paths.append(str(wanted_path)[:-3])
                else:
                    num_of_sub_sub_folder_skips += 1
                    print(f"Warning: {wanted_path} missing.")
                    
        print(f"Num_of_parent_skips: {num_of_parent_skips}")
        print(f"num_of_sub_folder_skips: {num_of_sub_folder_skips}")
        print(f"num_of_sub_sub_folder_skips: {num_of_sub_sub_folder_skips}")
        # if not parent_qualified:
        #     continue

    return collected_paths

def load_mesh(mesh_dir: str = "data/CDR/mesh_save_dir/rectangle.xdmf"):
    mesh = Mesh()
    with XDMFFile(mesh_dir) as xdmf:
        xdmf.read(mesh)
    return mesh
def load_function_space(mesh: Mesh, cg_order=1):
    V = FunctionSpace(mesh, 'CG', cg_order) 
    return V

def load_fenics_function(field_file_path: str, mesh_dir: str="data/CDR/mesh_save_dir/rectangle.xdmf"):
    """ this function takes in a path to the file where fenics function of the field_of_interest is located 
    and returns the associated "<class 'dolfin.function.function.Function'>", on a CG-1 elements. """
    field_of_interest = Path(field_file_path).name
    if field_of_interest not in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']:
        raise ValueError("cdr field must be one of 'fuel_field', 'oxygen_field', 'product_field', 'temp_field'")
    mesh = load_mesh(mesh_dir=mesh_dir)
    V_1 = FunctionSpace(mesh, 'CG', 1) 
    field_t_now = Function(V_1)
    with XDMFFile(mesh.mpi_comm(), field_file_path) as xdmf_1:
        xdmf_1.read_checkpoint(field_t_now, field_of_interest, 0)
    return field_t_now

def get_mesh_bounds(mesh):
    mesh_coords = mesh.coordinates()
    xmin, ymin = mesh_coords.min(axis=0)
    xmax, ymax = mesh_coords.max(axis=0)
    return (xmin, xmax), (ymin, ymax)

def sample_fenics_function(field_file_path: str, 
                        mesh: Mesh,
                        func_space_V: FunctionSpace,
                        test_domain: ndarray = np_array([[0,1],[0,0.5]]), 
                        num_of_spatial_sampling_m: int = 5,
                        g_constraint: float = None):
    m = num_of_spatial_sampling_m
    test_domain = np_array(test_domain)
    
    x_vect = np_unif([test_domain[0,0], test_domain[1,0]], [test_domain[0,1], test_domain[1,1]], size=(m, 2))
    field_of_interest = Path(field_file_path).name
    
    f = Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), field_file_path) as xdmf_1:
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

def approximate_set_lebesgue(binary_system_output_data: ndarray,
                            lambda_X: float) -> ndarray:
    
    _, m = binary_system_output_data.shape

    #row sums /w output shape (n,)
    just_ones_count = binary_system_output_data.sum(axis=1)

    #pairwise AND count /w output shape (n,n)
    and_count = binary_system_output_data @ binary_system_output_data.T

    #xOr count using identity AxOrB=a+b-2(aANDb)
    xOr_count = just_ones_count[:, None] + just_ones_count[None, :] - 2*and_count
    xOr_count = xOr_count.astype(float32)

    lambda_matrix = lambda_X*(xOr_count/m)
    
    return lambda_matrix

def get_K_gamma(field_data_dirs_list: list = None,
                n: int = None,
                num_of_spatial_sampling_m: int = None,
                mesh: Mesh = None,
                func_space_V: FunctionSpace = None,
                test_domain: ndarray = np_array([[0,1],[0,0.5]]),
                g_constraint: float = None,
                verbose: bool = False,
                binary_system_output_data = None):
    
    if binary_system_output_data is None:
        m = num_of_spatial_sampling_m
        binary_system_output_data = np_zeros((n, m), dtype=uint8)
        if verbose:
            t0_0 = time_ns()
        for i, dir in enumerate(field_data_dirs_list):
            binary_system_output_data[i] = sample_fenics_function(field_file_path=dir,
                                                    mesh=mesh,
                                                    func_space_V=func_space_V,
                                                    test_domain=test_domain,
                                                    num_of_spatial_sampling_m=m,
                                                    g_constraint=g_constraint)
        if verbose:
            t0_1 = time_ns()
    lambda_X = 1
    for i in range(len(test_domain)):
        lambda_curr_dom = max(test_domain[i])-min(test_domain[i])
        if lambda_curr_dom != 0:
            lambda_X *= lambda_curr_dom
    if verbose:
        t1_0 = time_ns()

    lambda_matrix = approximate_set_lebesgue(binary_system_output_data=binary_system_output_data, lambda_X=lambda_X)

    if verbose:
        t1_1 = time_ns()

    sigma_squared = np_mean(lambda_matrix)

    if verbose:
        t2_1 = time_ns()

    K_gamma = np_exp(-1.0*lambda_matrix/(2*sigma_squared))

    if verbose:
        t3_1 = time_ns()

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

def get_K_U_sobolev_looped(data_directory: str = None,
                    n: int = 5,
                    num_of_inputs: int = 5,
                    which_input_one_hot: str = '00001',
                    input_data=None,
                    verbose: bool = False):
    if verbose:
        t0 = time_ns()    

    dim_U = num_of_inputs
    if input_data is None:
        input_data_dirs_list = get_data_file_dirs(data_directory, data_type='input_data')
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
            K_ij = 1.0

            for d in active_dims:
                u_i = u_arr[i, d]
                u_j = u_arr[j, d]
                K_ij *= sobolev_kernel_univar(u_i, u_j)
            K_sob_matrix[i,j] = K_ij
            K_sob_matrix[j,i] = K_ij
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of get_K_U_sobolev_looped: {dt:.3f} (s)')
    return K_sob_matrix

def get_K_U_sobolev_vectorized(data_directory: str = None,
                    n: int = 5,
                    num_of_inputs: int = 5,
                    which_input_one_hot: str = '00001',
                    chunk_size=None,
                    input_data=None,
                    verbose:bool=False):
    if verbose:
        t0 = time_ns()
    dim_U = num_of_inputs
    if input_data is None:
        input_data_dirs_list = get_data_file_dirs(data_directory, data_type='input_data')
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

        U_i = u_active[start_idx:end_idx]
        u_i = U_i[:, :, None]
        u_j = u_active.T[None, :, :]
        du = u_i - u_j
        K_curr_chunk_sobolev = (1.0 + (u_i-0.5)*(u_j-0.5) + 0.5*(du*du-np_abs(du)+1.0/6.0))
        K_sob_matrix[start_idx:end_idx] = np_prod(K_curr_chunk_sobolev, axis=1)
        start_idx = end_idx
    
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of get_K_U_sobolev_vectorized with chunks of {chunk_size}: {dt:.3f} (s)')

    return K_sob_matrix

def calculate_hsic_looped(K_U: ndarray, K_gamma: ndarray, verbose: bool = False) -> float:
    if verbose:
        t0 = time_ns()
    n = K_U.shape[0]
    HSIC_fellmann = 0.0
    const_0 = 2.0 / (n * (n - 1))

    for j in range(1, n):
        for i in range(j):
            HSIC_fellmann += (K_U[i, j] - 1.0) * K_gamma[i, j]
    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of calculate_hsic_looped: {dt:.3f} (s)')
    return const_0 * HSIC_fellmann

def calculate_hsic_vectorized(K_U: ndarray, K_gamma: ndarray, verbose: bool = False) -> float:
    if verbose:
        t0 = time_ns()
    n = K_U.shape[0]
    const_0 = 2.0 / (n * (n - 1))

    #upper-triangle indices (i < j)
    up_tri_indices_i, up_tri_indices_j = np_triu_indices(n, k=1)

    HSIC_fellmann = (K_U[up_tri_indices_i, up_tri_indices_j] - 1.0) * K_gamma[up_tri_indices_i, up_tri_indices_j]

    if verbose:
        t1 = time_ns()
        dt = (t1-t0) / 1e9
        print(f'Wall-clock time of calculate_hsic_vectorized: {dt:.3f} (s)')
    return const_0 * np_sum(HSIC_fellmann)

def transform_logUnif_to_unitUnif(min_u, max_u, log_unif_samples):
    transformed_samples = (np_log(log_unif_samples) - np_log(min_u))/(np_log(max_u)-np_log(min_u))
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

def transform_all_u_inputs(u_arr: ndarray, u_domain_specs: list):
    n, d = u_arr.shape
    assert d == len(u_domain_specs)

    u_arr_transformed = np_zeros_like(u_arr, dtype=float)

    for u_idx in range(d):
        u_spec = u_domain_specs[u_idx]
        u_curr_idx = u_arr[:, u_idx]
        min_u = u_spec['min']
        max_u = u_spec['max']

        if u_spec['distribution_type'] == 'log_uniform':
            u_arr_transformed[:, u_idx] = transform_logUnif_to_unitUnif(min_u, max_u, u_curr_idx)
        elif u_spec['distribution_type'] == 'uniform':
            u_arr_transformed[:, u_idx] = transform_unif_to_unitUnif(min_u, max_u, u_curr_idx)
        else:
            raise ValueError(f"Unknown distribution type '{u_spec['distribution_type']}' for dimension {u_idx}.") 
    return u_arr_transformed

def get_u_index_superset_one_hot_binstrs(dim_of_U, higher_order=False):
    return [flipStr(i) for i in sorted(getIndexSuperset(dim_of_U, higher_order=higher_order), key=lambda x: directBinStrSum(x))]
