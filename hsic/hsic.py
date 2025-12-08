from numpy import ndarray, array as np_array
from numpy import zeros as np_zeros
from numpy import load as np_load
from numpy import arange as np_arange
from numpy.random import shuffle as np_shuffle
from kbsa.hsic_utils import (get_data_file_dirs, 
                        transform_all_u_inputs, 
                        get_u_index_superset_one_hot_binstrs,
                        get_K_U_sobolev_vectorized,
                        load_mesh,
                        load_function_space,
                        get_K_gamma,
                        calculate_hsic_vectorized)
def hsic(data_directory: str,
        mesh_directory: str = 'data/CDR/mesh_save_dir/rectangle.xdmf',
        field_of_interest: str = 'temp_field',
        n: int = 10, 
        num_of_spatial_sampling_m: int = None,
        test_domain: ndarray = np_array([[0,1],[0,0.5]]),
        u_domain_specs: list = [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                                {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                                {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                                {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                                {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}],
        u_one_hot_key_map: dict = {'10000': 'A', 
                                '01000': 'E',
                                '00100': 'T_i',
                                '00010': 'T_o',
                                '00001': 'phi'},
        which_input_one_hot: str = None,
        g_constraint: float = None,
        shuffle_inputs: bool = False,
        chunk_size=None,
        binary_system_output_data=None,
        input_data_dirs_to_use_parall_processed=None,
        output_data_type: str='fenics_function'):

    # test_domain = np_array(test_domain)
    num_of_u_inputs = len(u_domain_specs)

    if input_data_dirs_to_use_parall_processed is None:
        input_data_dirs_list = get_data_file_dirs(data_directory, data_type='input_data')

        n_max = len(input_data_dirs_list)
        data_dir_indices = np_arange(0, n_max)
        if shuffle_inputs:
            np_shuffle(data_dir_indices)
            
        n_max = len(input_data_dirs_list)
        if n < n_max:
            data_dir_indices = data_dir_indices[:n]
        else:
            n = n_max

        input_data_dirs_list_to_use = [input_data_dirs_list[i] for i in data_dir_indices[:n]]
        field_data_dirs_list_to_use = [dir.replace('input_data.npy', field_of_interest) for dir in input_data_dirs_list_to_use]
    else:
        input_data_dirs_list_to_use = input_data_dirs_to_use_parall_processed
        
    print(f'Using {len(input_data_dirs_list_to_use)} data points.')
    #load input data
    u_arr = np_zeros((n, num_of_u_inputs))
    for i, dir in enumerate(input_data_dirs_list_to_use):
        u_arr[i] = np_load(dir)
    u_arr_transformed = transform_all_u_inputs(u_arr=u_arr, u_domain_specs=u_domain_specs)
    #maybe a heuristic for choosing chunk sizes?
    if chunk_size is None or chunk_size <= 0:
        chunk_size = int(n/5)
    
    inputs_one_hot_binstrs_list = get_u_index_superset_one_hot_binstrs(dim_of_U=num_of_u_inputs)
    if which_input_one_hot is None:
        inputs_one_hot_binstrs_list_to_use = inputs_one_hot_binstrs_list
    else:
        if which_input_one_hot in inputs_one_hot_binstrs_list:
            inputs_one_hot_binstrs_list_to_use = [which_input_one_hot, inputs_one_hot_binstrs_list[-1]]
        else:
            raise ValueError(f"Invalid which_input_one_hot: {which_input_one_hot} is not in {inputs_one_hot_binstrs_list}.")

    K_U_dict = {}
    for key in inputs_one_hot_binstrs_list_to_use:
        K_U_dict[key] = get_K_U_sobolev_vectorized(input_data=u_arr_transformed,
                                        n=n, 
                                        num_of_inputs=num_of_u_inputs,
                                        which_input_one_hot=key, 
                                        chunk_size=chunk_size,
                                        verbose=False)
    # for key in inputs_one_hot_binstrs_list_to_use[:-1]:
    #     K_U_dict[key] = K_U_dict[key]/K_U_dict[inputs_one_hot_binstrs_list_to_use[-1]]


    if num_of_spatial_sampling_m is None:
        m = n
    else:
        m = num_of_spatial_sampling_m
    if binary_system_output_data is None:
        my_mesh = load_mesh(mesh_dir=mesh_directory)
        V = load_function_space(my_mesh)
        K_gamma = get_K_gamma(field_data_dirs_list=field_data_dirs_list_to_use,
                        n=n,
                        num_of_spatial_sampling_m=m,
                        mesh=my_mesh,
                        func_space_V=V,
                        test_domain=test_domain,
                        g_constraint=g_constraint)
    else:
        K_gamma = get_K_gamma(binary_system_output_data=binary_system_output_data,
                            test_domain=test_domain)
    
    hsic_vals_dict = {}

    u_all_bitstr = inputs_one_hot_binstrs_list_to_use[-1]
    hsic_all = calculate_hsic_vectorized(K_U=K_U_dict[u_all_bitstr], K_gamma=K_gamma, verbose=False)
    for key in inputs_one_hot_binstrs_list_to_use[:-1]:
        hsic_vals_dict[key] = calculate_hsic_vectorized(K_U=K_U_dict[key], K_gamma=K_gamma, verbose=False)/hsic_all
    
    hsic_vals_dict_key_mapped = {u_one_hot_key_map[key]: val for key, val in hsic_vals_dict.items()}
    return hsic_vals_dict_key_mapped