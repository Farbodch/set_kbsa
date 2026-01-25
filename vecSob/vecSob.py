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
                                              get_index_complement,
                                              direct_binstr_sum)
from numeric_models.analytic_models import ishigami_vectorized_generator as gen_ishigami
from numba.core.registry import CPUDispatcher
from types import FunctionType
from dolfin import MPI as dolfin_MPI
from mpi4py import MPI as pyMPI
""" 
LEFT TO DO:
    Read-in data from file & sampling Both with-MPI and without-MPI
    Test-cases (most important -> consistancy between einsumed, vectorized, and looped.)
"""
def vecSob(data_directory: str | None = None,
        n: int = 10,
        num_of_grid_points: int | None = None,
        test_domain: NDArray | None = np_array([[0,1],[0,0.5]]),
        map_one_hot_keys: bool = True,
        u_one_hot_key_map: dict | None = {'10000': 'A', 
                                    '01000': 'E',
                                    '00100': 'T_i',
                                    '00010': 'T_o',
                                    '00001': 'phi'},
        #if which_input_one_hot passed in is A='11110' and return return_as_total_vecSobs is set to True,
        #then we would return total sob index of complement of '11110', which is A_c='00001'
        return_total_vecSobs: bool = False, 
        num_of_u_inputs: int | None = None,
        which_input_one_hot: str | None = None,
        g_constraint: float | None = None, 
        shuffle_inputs: bool = False,
        chunk_size: int | None = None,
        #this is to contain the {y_I<=g_constraint}, {y_II<=g_constraint}, 
        # and {y_tilde<=g_constraint}, where y \in R^n*h where h is the num_of_grid_points and n is 
        # the sampling number. {y_i<=g_constraint}\in{0,1} and binary_system_output_data_dict.shape = (3, n, h)
        binary_system_output_data_dict: dict | None = None,
        process_all_order_one_indices: bool = False, #maybe instead of this, can send in which_order numbers instead?
        input_data_dirs_to_use_parall_processed=None,
        process_type: str = 'fenics_function',
        fem_process_settings: dict={'fem_mesh_directory': 'data/CDR/mesh_save_dir/rectangle.xdmf',
                                    'field_of_interest': 'temp_field'},
        analytical_process_settings: dict = {'process_generator': gen_ishigami}):

    #this exists for reading in input-data within vecSob(). Not implemented yet.
    #is it even necessary? Better to separate data-read-in and organization?
    if process_type == 'fenics_function' and binary_system_output_data_dict is None:
        fem_mesh_directory = fem_process_settings['fem_mesh_directory']
        field_of_interest = fem_process_settings['field_of_interest']
    
    #this is a check to ensure this function has the information of how many random-inputs the 
    #process model has.
    if num_of_u_inputs is None and which_input_one_hot is not None:
        num_of_u_inputs = len(which_input_one_hot)
    elif num_of_u_inputs is None and u_one_hot_key_map is not None:
        num_of_u_inputs = len(list(u_one_hot_key_map.keys())[0])
    elif num_of_u_inputs is None and which_input_one_hot is None and u_one_hot_key_map is None:
        raise ValueError('Missing information for which_input_one_hot, u_one_hot_key_map, and num_of_u_inputs.')
    if process_all_order_one_indices:
        index_set = [idx for idx in generator_order_r_idcs_as_onehot(1, num_of_u_inputs)]
        # u_one_hot_key_map_t = {get_index_complement(key): f"{val}_t" for key, val in u_one_hot_key_map.items()}
        # u_one_hot_key_map.update(u_one_hot_key_map_t)
    else:
        index_set = [which_input_one_hot]
        try:
            binary_system_output_data_dict.keys()
        except Exception as e:
            print(binary_system_output_data_dict)
            raise ValueError(str(e)) from e
        assert which_input_one_hot in list(binary_system_output_data_dict.keys()), \
            f"The passed in which_input_one_hot {which_input_one_hot}, is not a key of binary_system_output_data_dict:{list(binary_system_output_data_dict.keys())}."
    """
    binary_system_output_data_dict = {index_str: binary_system_output_data_A}
    binary_system_output_data_dict[key].shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
    if binary_system_output_data_dict is None: #then call sample_fenics_function()? This needs to be MPI-accelerated.
        # pass #temp
        raise ValueError("binary_system_output_data_dict must be passed-in!") #CHANGE THIS!!
    
    vecSob_vals_dict = {}
    for index in index_set:
        vecSob_vals_dict[index] = calculate_vecSob_index_A_einsumed(binary_system_output_data_index_A=binary_system_output_data_dict[index])
    
    if map_one_hot_keys:
        vecSob_vals_dict_key_mapped = {}
        for key, val in vecSob_vals_dict.items():
            if key in u_one_hot_key_map.keys():
                vecSob_vals_dict_key_mapped[u_one_hot_key_map[key]] = val
            else:
                vecSob_vals_dict_key_mapped[key] = val
        vecSob_vals_dict = vecSob_vals_dict_key_mapped

    # vecSob_vals_dict_key_mapped = {u_one_hot_key_map[key]: val for key, val in vecSob_vals_dict.items()}
    #if return_total_vecSobs is True, then binary_system_output_data_dict must contain atleast TWO members, one index A and its complement A_c
    if return_total_vecSobs:
        t_vecSob_vals_dict = {}
        #true if we indicated that we want vecSob() to only return the result of one index A instead of a set of them.
        if process_all_order_one_indices:
            t_index_set = [idx for idx in generator_order_r_idcs_as_onehot(num_of_u_inputs-1, num_of_u_inputs)]
        else:
            t_index_set = [get_index_complement(which_input_one_hot)]
        for index in t_index_set:
            t_vecSob_vals_dict[index] = 1-calculate_vecSob_index_A_einsumed(binary_system_output_data_index_A=binary_system_output_data_dict[index])
        if map_one_hot_keys:
            t_vecSob_vals_dict_key_mapped = {}
            for key, val in t_vecSob_vals_dict.items():
                if key in u_one_hot_key_map.keys():
                    t_vecSob_vals_dict_key_mapped[f"t_{u_one_hot_key_map[get_index_complement(key)]}"] = val
                else:
                    t_vecSob_vals_dict_key_mapped[f"t_{get_index_complement(key)}"] = val
            t_vecSob_vals_dict = t_vecSob_vals_dict_key_mapped

        return vecSob_vals_dict, t_vecSob_vals_dict
    return vecSob_vals_dict

#not properly supporting orders higher than 1 yet.
def vecSob_parallelized(comm, 
                        sorted_output_data_dict: dict,
                        num_of_u_inputs: int,
                        u_one_hot_key_map: dict,
                        get_total_sobols: bool,
                        which_orders: list):

    rank = comm.Get_rank()
    size = comm.Get_size()
    TAG_WORK = 1
    TAG_DONE = 2
    TAG_STOP = 3

    # Root-branch: work distribution
    if rank == 0:
        jobs = list(sorted_output_data_dict.items())
        P = len(jobs) #num of indices to be calculated

        next_job = 0
        active_workers = 0

        #dispatch jobs to as many workers as are available
        for r in range(1, size):
            if next_job < P:
                comm.send(jobs[next_job], dest=r, tag=TAG_WORK)
                next_job += 1
                active_workers += 1
            else:
                comm.send(None, dest=r, tag=TAG_STOP)

        vecSob_results = {}

        #dynamic scheduling between idle workers and 
        while active_workers > 0:
            # receive results
            result_dict = comm.recv(source=pyMPI.ANY_SOURCE, tag=TAG_DONE)
            idle_worker_rank = result_dict['_rank']
            result_dict.pop('_rank')
            vecSob_results.update(result_dict)

            # dispatch one job to a newly idled worker, if any jobs left
            if next_job < P:
                comm.send(jobs[next_job], dest=idle_worker_rank, tag=TAG_WORK)
                next_job += 1

            # stop worker if no more jobs left
            else:
                comm.send(None, dest=idle_worker_rank, tag=TAG_DONE)
                active_workers -= 1

        # send one job to each worker
        # for r in range(1, min(size, P + 1)):
        #     comm.send(jobs[r - 1], 
        #               dest=r, 
        #               tag=TAG_WORK)
        
        # for r in range(P + 1, size):
        #     comm.send(None, 
        #               dest=r,
        #               tag=TAG_WORK)
        
        
        # for _ in range(min(P, size - 1)):
        #     result_dict = comm.recv(source=pyMPI.ANY_SOURCE, 
        #                             tag=TAG_DONE)
        #     vecSob_results.update(result_dict)

        # # write once
        # with open(f"{parent_folder}/results.txt", "w") as f:
        #     f.write(f"{vecSob_results}")
        vecSob_results_mapped = {}
        for key, val in vecSob_results.items():
            if direct_binstr_sum(key) == 1:
                vecSob_results_mapped[u_one_hot_key_map[key]] = val
            else:
                if direct_binstr_sum(key) in which_orders:
                    vecSob_results_mapped[key] = val
                if get_total_sobols and direct_binstr_sum(key) == (num_of_u_inputs-1):
                    vecSob_results_mapped[f"t_{u_one_hot_key_map[get_index_complement(key)]}"] = 1-val

        return vecSob_results_mapped   
    
    # Worker-branch: perform the given work
    else:
        while True:
            recieved_data = comm.recv(source=0, tag=pyMPI.ANY_TAG)
            if recieved_data is None:
                break
            key, data = recieved_data
            idx_data_dict = {key: data}
            try:
                result_dict = vecSob(u_one_hot_key_map=None,
                                     map_one_hot_keys=False,
                                     which_input_one_hot=key,
                                     num_of_u_inputs=num_of_u_inputs,
                                     binary_system_output_data_dict=idx_data_dict,
                                     process_all_order_one_indices=False,
                                     return_total_vecSobs=False)
            except Exception as e:
                result_dict = {f"ERROR_{key}": {"repr": repr(e), "message": str(e)}}
            result_dict["_rank"] = rank
            #send the result back to Root-branch for collection
            comm.send(result_dict, dest=0, tag=TAG_DONE)

        # idx_data_dict = comm.recv(source=0, tag=TAG_WORK)
        # if idx_data_dict is not None:
        #     key, data = idx_data_dict
        #     idx_data_dict = {key: data}
        #     try:
        #         result_dict = vecSob(u_one_hot_key_map=None,
        #                              map_one_hot_keys=False,
        #                              which_input_one_hot=key,
        #                              num_of_u_inputs=num_of_u_inputs,
        #                              binary_system_output_data_dict=idx_data_dict,
        #                              process_all_order_one_indices=False,
        #                              return_total_vecSobs=False)
        #     except Exception as e:
        #         result_dict = {f"ERROR_{key}": repr(e)}
        #         raise ValueError(str(e)) from e
        #     #send the result back to Root-branch for collection
        #     comm.send(result_dict, 
        #               dest=0, 
        #               tag=TAG_DONE)
        return None