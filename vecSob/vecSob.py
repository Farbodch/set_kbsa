from numpy import (array as np_array,
                asarray as np_asarray,
                einsum as np_einsum,
                float64 as np_float64,
                log2 as np_log2,
                floor as np_floor)
from numpy.typing import NDArray
from vecSob.vecSob_utils import (sample_fenics_function, 
                                 calculate_vecSob_index_A_einsumed,
                                 calculate_vecSob_index_A_vectorized,
                                 calculate_vecSob_index_A_looped)
from auxiliary_utils.io_management import (get_input_data_from_file_analytical,
                                            get_input_data_from_file_fenics_function,
                                            load_mesh,
                                            load_function_space)
from auxiliary_utils.index_management import (generator_order_r_idcs_as_onehot,
                                              get_index_complement)
from numeric_models.analytic_models import ishigami_vectorized_generator as gen_ishigami
from numba.core.registry import CPUDispatcher
from types import FunctionType

def vecSob(data_directory: str | None = None,
        n: int = 10,
        num_of_grid_points: int | None = None,
        test_domain: NDArray | None = np_array([[0,1],[0,0.5]]),
        u_one_hot_key_map: dict = {'10000': 'A', 
                                    '01000': 'E',
                                    '00100': 'T_i',
                                    '00010': 'T_o',
                                    '00001': 'phi'},
        return_total_vecSobs: bool = False,
        num_of_u_inputs: int | None = None,
        which_input_one_hot: str | None = None,
        g_constraint: float | None = None, 
        shuffle_inputs: bool = False,
        chunk_size: int | None = None,
        #this is to contain the {y_I<=g_constraint}, {y_II<=g_constraint}, 
        # and {y_tilde<=g_constraint}, where y \in R^n*h where h is the num_of_grid_points and n is 
        # the sampling number. {y_i<=g_constraint}\in{0,1} and binary_system_output_data.shape = (3, n, h)
        binary_system_output_data: dict | None = None,
        input_data_dirs_to_use_parall_processed=None,
        process_type: str = 'fenics_function',
        fem_process_settings: dict={'fem_mesh_directory': 'data/CDR/mesh_save_dir/rectangle.xdmf',
                                    'field_of_interest': 'temp_field'},
        analytical_process_settings: dict = {'process_generator': gen_ishigami}):

    if process_type == 'fenics_function':
        fem_mesh_directory = fem_process_settings['fem_mesh_directory']
        field_of_interest = fem_process_settings['field_of_interest']
    
    if num_of_u_inputs is None and which_input_one_hot is not None:
        num_of_u_inputs = len(which_input_one_hot)
    elif num_of_u_inputs is None and u_one_hot_key_map is not None:
        num_of_u_inputs = len(list(u_one_hot_key_map.keys())[0])
    else:
        raise ValueError('Missing information for which_input_one_hot, u_one_hot_key_map, and num_of_u_inputs.')

    index_set = [idx for idx in generator_order_r_idcs_as_onehot(1,num_of_u_inputs)]
        # u_one_hot_key_map_t = {get_index_complement(key): f"{val}_t" for key, val in u_one_hot_key_map.items()}
        # u_one_hot_key_map.update(u_one_hot_key_map_t)
    """
    binary_system_output_data = {index_str: binary_system_output_data_A}
    binary_system_output_data[key].shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
    if binary_system_output_data is None: #then call sample_fenics_function()? This needs to be MPI-accelerated.
        # pass #temp
        raise ValueError("binary_system_output_data must be passed-in!") #CHANGE THIS!!
    
    vecSob_vals_dict = {}
    for index in index_set:
        vecSob_vals_dict[index] = calculate_vecSob_index_A_einsumed(binary_system_output_data_index_A=binary_system_output_data[index])
    vecSob_vals_dict_key_mapped = {u_one_hot_key_map[key]: val for key, val in vecSob_vals_dict.items()}
    if return_total_vecSobs:
        t_index_set = [idx for idx in generator_order_r_idcs_as_onehot(num_of_u_inputs-1,num_of_u_inputs)]
        t_vecSob_vals_dict = {}
        for index in t_index_set:
            t_vecSob_vals_dict[index] = 1-calculate_vecSob_index_A_einsumed(binary_system_output_data_index_A=binary_system_output_data[index])
        t_vecSob_vals_dict_key_mapped = {f"t_{u_one_hot_key_map[get_index_complement(key)]}": val for key, val in t_vecSob_vals_dict.items()}
        return vecSob_vals_dict_key_mapped, t_vecSob_vals_dict_key_mapped
    return vecSob_vals_dict_key_mapped