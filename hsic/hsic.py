from numpy import (array as np_array,
                log2 as np_log2,
                floor as np_floor)
from numpy.typing import NDArray
from hsic.hsic_utils import (transform_all_u_inputs, 
                            get_u_index_superset_one_hot_binstrs,
                            get_K_gamma,
                            get_K_U_sobolev_vectorized,
                            get_K_U_sobolev_looped,
                            calculate_hsic_vectorized,
                            calculate_hsic_looped,
                            calculate_hsic_traced)
from auxiliary_utils.io_management import (get_input_data_from_file_analytical,
                                            get_input_data_from_file_fenics_function,
                                            load_mesh,
                                            load_function_space)
from numeric_models.analytic_models import ishigami_vectorized_generator as gen_ishigami
from numba.core.registry import CPUDispatcher
from types import FunctionType

def hsic(data_directory: str | None = None,
        n: int = 10, 
        num_of_spatial_sampling_m: int | None = None,
        test_domain: NDArray = np_array([[0,1],[0,0.5]]),
        u_domain_specifications: list = [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                                {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                                {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                                {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                                {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}],
        u_one_hot_key_map: dict = {'10000': 'A', 
                                '01000': 'E',
                                '00100': 'T_i',
                                '00010': 'T_o',
                                '00001': 'phi'},
        which_input_one_hot: str | None = None,
        g_constraint: float | None = None,
        shuffle_inputs: bool = False,
        chunk_size: int | None = None,
        binary_system_output_data: NDArray | None = None,
        input_data_dirs_to_use_parall_processed=None,
        u_arr: NDArray | None = None,
        process_type: str = 'fenics_function',
        fem_process_settings: dict={'fem_mesh_directory': 'data/CDR/mesh_save_dir/rectangle.xdmf',
                                    'field_of_interest': 'temp_field'},
        analytical_process_settings: dict = {'process_generator': gen_ishigami},
        vectorized_hsic_flag: bool = True,
        vectorized_K_U_flag: bool = True,
        hsic_control_method_flag: bool = False,
        verbose_K_gamma: bool = False):
    
    if process_type == 'fenics_function':
        fem_mesh_directory = fem_process_settings['fem_mesh_directory']
        field_of_interest = fem_process_settings['field_of_interest']
    if process_type == 'analytical':
        process_generator = analytical_process_settings['process_generator']
        if not isinstance(process_generator, (FunctionType, CPUDispatcher)):
            raise TypeError(f"Expected a Numba @njit function or a Python function for process_generator, got {type(process_generator)}!")
    num_of_u_inputs = len(u_domain_specifications)

    #load-in input data
    if u_arr is None:
        """CAN REFACTOR THIS PART, AND THE FUNCTIONS -> Make cleaner!"""
        if process_type == 'fenics_function':
            u_arr, input_data_dirs_list_to_use = get_input_data_from_file_fenics_function(data_directory=data_directory,
                                                                                    num_of_u_inputs=num_of_u_inputs,
                                                                                    n=n,
                                                                                    shuffle=shuffle_inputs,
                                                                                    return_directories=True,
                                                                                    input_data_directories_list_to_use=input_data_dirs_to_use_parall_processed) 
            # input_data_dirs_list = get_data_file_dirs(data_directory, data_type='input_data')
            # n_max = len(input_data_dirs_list)
            # data_dir_indices = np_arange(0, n_max)
            # if n < n_max:
            #     data_dir_indices = data_dir_indices[:n]
            # else:
            #     n = n_max

            # if shuffle_inputs:
            #     np_shuffle(data_dir_indices)
            
            #if shuffle_inputs was toggled, then we shuffle based on indices and then add the data file directories here
            # input_data_dirs_list_to_use = [input_data_dirs_list[i] for i in data_dir_indices[:n]]
            data_dirs_to_eval_list = [dir.replace('input_data.npy', field_of_interest) for dir in input_data_dirs_list_to_use]
        elif process_type == 'analytical':
            u_arr = get_input_data_from_file_analytical(data_directory=data_directory,
                                                        num_of_u_inputs=num_of_u_inputs,
                                                        n=n,
                                                        return_directories=False)
    n = u_arr.shape[0]
    print(f'Using {n} data points.')
    u_arr_transformed = transform_all_u_inputs(u_arr=u_arr, u_domain_specifications=u_domain_specifications)
    #Choosing chunk size as the 2 to the power of closest power of two minus 2.
    if chunk_size is None or chunk_size <= 0:
        chunk_size = 2**int(np_floor(np_log2(n))-2)
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
        if vectorized_K_U_flag:
            K_U_dict[key] = get_K_U_sobolev_vectorized(input_data=u_arr_transformed,
                                            n=n, 
                                            num_of_inputs=num_of_u_inputs,
                                            which_input_one_hot=key, 
                                            chunk_size=chunk_size,
                                            verbose=True)
        else:
            K_U_dict[key] = get_K_U_sobolev_looped(input_data=u_arr_transformed,
                                            n=n, 
                                            num_of_inputs=num_of_u_inputs,
                                            which_input_one_hot=key,
                                            verbose=False)
    if num_of_spatial_sampling_m is None:
        m = n
    else:
        m = num_of_spatial_sampling_m
    if binary_system_output_data is None:
        if process_type == 'fenics_function':
            my_mesh = load_mesh(mesh_dir=fem_mesh_directory)
            V = load_function_space(my_mesh)
            process_generator = None
            u_arr = None
        else:
            my_mesh = None
            V = None
        K_gamma = get_K_gamma(process_type=process_type,
                            data_dirs_to_eval_list=data_dirs_to_eval_list,
                            n=n,
                            num_of_spatial_sampling_m=m,
                            mesh=my_mesh,
                            func_space_V=V,
                            test_domain=test_domain,
                            g_constraint=g_constraint,
                            verbose=verbose_K_gamma,
                            process_generator=process_generator,
                            u_arr=u_arr)

    else:
        K_gamma = get_K_gamma(binary_system_output_data=binary_system_output_data,
                            test_domain=test_domain)
        
    hsic_vals_dict = {}

    u_all_bitstr = inputs_one_hot_binstrs_list_to_use[-1]
    if hsic_control_method_flag:
        hsic_all = calculate_hsic_traced(K_U=K_U_dict[u_all_bitstr], K_gamma=K_gamma, verbose=True)
    else:
        if vectorized_hsic_flag:
            hsic_all = calculate_hsic_vectorized(K_U=K_U_dict[u_all_bitstr], K_gamma=K_gamma, verbose=True)
        else:
            hsic_all = calculate_hsic_looped(K_U=K_U_dict[u_all_bitstr], K_gamma=K_gamma, verbose=True)
    for key in inputs_one_hot_binstrs_list_to_use[:-1]:
        if hsic_control_method_flag:
            hsic_vals_dict[key] = calculate_hsic_traced(K_U=K_U_dict[key], K_gamma=K_gamma, verbose=True)/hsic_all
        else:
            if vectorized_hsic_flag:
                hsic_vals_dict[key] = calculate_hsic_vectorized(K_U=K_U_dict[key], K_gamma=K_gamma, verbose=True)/hsic_all
            else:
                hsic_vals_dict[key] = calculate_hsic_looped(K_U=K_U_dict[key], K_gamma=K_gamma, verbose=True)/hsic_all
    hsic_vals_dict_key_mapped = {u_one_hot_key_map[key]: val for key, val in hsic_vals_dict.items()}
    return hsic_vals_dict_key_mapped

