from numpy import (uint8, 
                   array as np_array,
                   zeros as np_zeros)
                #    arange as np_arange)
# from numpy.random import shuffle as np_shuffle
from vecSob.vecSob import vecSob, vecSob_parallelized
from auxiliary_utils.io_management import (get_data_file_directories,
                                        load_mesh,
                                        load_function_space)
from auxiliary_utils.index_management import generator_order_r_idcs_as_onehot
from vecSob.vecSob_utils import sample_fenics_function, build_sorted_output_data_dict
# from auxiliary_utils.mpi_management import adjust_sampling_number_for_hsic
from dolfin import MPI as dolfin_MPI
from datetime import datetime
from time import time as timetime
from uuid import uuid4
from os import makedirs

def main():
    comm = dolfin_MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    valid_cdr_fields = ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']
    process_model_name = '2d_cdr'
    process_settings={'2d_cdr': {'data_directory': 'data/experiment_data/cdr/vecSob',
                                'fem_mesh_directory': 'data/CDR/mesh_save_dir/rectangle.xdmf',
                                'u_domain_specifications': [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                                                            {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                                                            {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                                                            {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                                                            {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}],
                                'u_one_hot_key_map': {'10000': 'A', '01000': 'E', '00100': 'T_i', '00010': 'T_o', '00001': 'phi'},
                                'test_domain': np_array([[0.5, 0.15],[0.2, 0.3]]),
                                'g_constraint': 700,
                                'field_of_interest': valid_cdr_fields[3],
                                'which_orders': [1],
                                'get_total_sobols': True}}
    
    n = 10000
    num_of_grid_points = 20164 #should be power of the num of dimensions
    shuffle_inputs = True
    parallelize_flag = True
    binary_system_output_data = None
    data_file_dirs_dict = None
    # field_data_dirs_list_to_use = None

    """
    binary_system_output_data = {index_str: binary_system_output_data_A}
    binary_system_output_data[key].shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """

    if parallelize_flag:
        if rank == 0:
            num_of_u_inputs = len(process_settings[process_model_name]['u_domain_specifications'])
            
            #get all relevant directories
            data_files_dirs_list_bag = get_data_file_directories(base_dir=process_settings[process_model_name]['data_directory'], 
                                                        data_type='input_data',
                                                        process_type='cdr_vecSob',
                                                        verbose=False,
                                                        enforce_params=False)
            #adjust target file names to the fenics function field of interest
            field_data_dirs_list_bag = [dir.replace('input_data.npy', process_settings[process_model_name]['field_of_interest']) for dir in data_files_dirs_list_bag]
            
            #create dictionary to bin/organize the directories 
            data_file_dirs_dict = {'u_I': [], 'u_II': []}
            # n_maxs_dict = {'u_I': 0, 'u_II': 0}
            for r in process_settings[process_model_name]['which_orders']:
                for A_str in generator_order_r_idcs_as_onehot(r=r, d=num_of_u_inputs):
                    data_file_dirs_dict[A_str] = []
                    # n_maxs_dict[A_str] = 0
            if (num_of_u_inputs-1) not in process_settings[process_model_name]['which_orders'] and \
            process_settings[process_model_name]['get_total_sobols']:
                for A_str in generator_order_r_idcs_as_onehot(r=(num_of_u_inputs-1), d=num_of_u_inputs):
                    data_file_dirs_dict[A_str] = []
                    # n_maxs_dict[A_str] = 0
            #bin/organize the directories into the dictionary
            for dir in field_data_dirs_list_bag:
                for key in data_file_dirs_dict.keys():
                    if f'{key}/' in dir:
                        data_file_dirs_dict[key].append(dir)
                        # n_maxs_dict[A_str] += 1
            n_maxs_dict = {key: len(val) for key, val in data_file_dirs_dict.items()}
            #find the largest number of data points that all A_str share, n_max_shared
            n_max_shared = n_maxs_dict['u_I']
            n_max_any = n_maxs_dict['u_I']
            for key in n_maxs_dict.keys():
                if n_maxs_dict[key] < n_max_shared:
                    n_max_shared = n_maxs_dict[key]
                if n_maxs_dict[key] > n_max_any:
                    n_max_any = n_maxs_dict[key]
            #if the requested number of data points is less than n_max, then 
            if n < n_max_shared:
                for key in data_file_dirs_dict.keys():
                    data_file_dirs_dict[key] = data_file_dirs_dict[key][:n]
            elif n_max_shared != n_max_any:
                for key in data_file_dirs_dict.keys():
                    data_file_dirs_dict[key] = data_file_dirs_dict[key][:n_max_shared]
                n = n_max_shared
            else:
                n = n_max_shared
            print(f'Using {n} total data points per input index.')

        """ 
        CURRENTLY HERE -> 
            Now that we have all the directories organized into this dictionary (data_file_dirs_dict),
            X-we need to first clip them to have max n dirs in each bin, then bcast it into the different nodes,-X
            then we need distribute these dirs between all the nodes. Maybe each node can pick a combination of 
            directory and key from data_file_dirs_dict? How to orchestrate this between "size"-number of nodes?
            Maybe we do this block:
            --
            for i in indices:
                binary_system_output_data_local[i] = sample_fenics_function(data_directory=field_data_dirs_list_to_use[i],
                                                                            mesh=my_mesh,
                                                                            func_space_V=V,
                                                                            test_domain=process_settings[process_model_name]['test_domain'],
                                                                            num_of_spatial_sampling_m=m,
                                                                            g_constraint=process_settings[process_model_name]['g_constraint'])
            local_results = [(i, binary_system_output_data_local[i].copy()) for i in indices]
            --
            but for each key, and then sync and barrier()
        """

        data_file_dirs_dict = comm.bcast(data_file_dirs_dict, root=0)

        n = comm.bcast(n, root=0)
        max_jobs = (n + size - 1) // size 
        indices = range(rank, n, size)
        num_of_padded_runs_in_curr_rank = 0
        while len(indices) < max_jobs: #FAULTY WHILE LOGIC.
            num_of_padded_runs_in_curr_rank += 1
        assert num_of_padded_runs_in_curr_rank==0
        
        my_mesh = load_mesh(mesh_dir=process_settings[process_model_name]['fem_mesh_directory'])
        V = load_function_space(my_mesh)

        binary_system_output_data_dict = {}
        #loop through u_I, u_II, 00001, 00010, etc...
        for key in data_file_dirs_dict.keys():
            binary_system_output_data_local = np_zeros((n, num_of_grid_points), dtype=uint8)
            #here we distribute data read-in among available nodes.
            for i in indices:
                binary_system_output_data_local[i] = sample_fenics_function(data_directory=data_file_dirs_dict[key][i],
                                                                            mesh=my_mesh,
                                                                            func_space_V=V,
                                                                            test_domain=process_settings[process_model_name]['test_domain'],
                                                                            g_constraint=process_settings[process_model_name]['g_constraint'],
                                                                            num_of_grid_points=num_of_grid_points)
            local_results_curr_key = [(i, binary_system_output_data_local[i].copy()) for i in indices]
            all_results_curr_key = comm.gather(local_results_curr_key, root=0)
            comm.barrier()
            if rank==0:
                binary_system_output_data = np_zeros((n, num_of_grid_points), dtype=uint8)
                for rank_list in all_results_curr_key:
                    for i, results in rank_list:
                        binary_system_output_data[i] = results
                binary_system_output_data_dict[key] = binary_system_output_data
        comm.barrier()

        if rank == 0:
            sorted_output_data_dict = build_sorted_output_data_dict(binary_system_output_data_dict)
        else:
            sorted_output_data_dict = None
        vecSob_results_mapped = vecSob_parallelized(comm=comm,
                                             sorted_output_data_dict=sorted_output_data_dict,
                                             num_of_u_inputs=len(process_settings[process_model_name]['u_domain_specifications']),
                                             u_one_hot_key_map=process_settings[process_model_name]['u_one_hot_key_map'],
                                             get_total_sobols=process_settings[process_model_name]['get_total_sobols'],
                                             which_orders=process_settings[process_model_name]['which_orders'])

    if rank==0:
        parent_uid = datetime.now().strftime("%Y_%m_%d_%H_%M_%S__") + str(uuid4().hex)
        parent_folder = f"data/vecSob_results/{parent_uid}"
        makedirs(parent_folder, exist_ok=True)
        with open(f"{parent_folder}/meta_data.txt", 'w') as f:
                field_of_interest = process_settings[process_model_name]['field_of_interest']
                test_domain = process_settings[process_model_name]['test_domain']
                g_constraint = process_settings[process_model_name]['g_constraint']
                f.write(f'parent_uid_{parent_uid};\nnum_of_outer_loop_sampling_n:{n};\nnum_of_grid_points{num_of_grid_points};\nfield_of_interest:{field_of_interest};\ntest_domain:{test_domain};\ng_constraint:{g_constraint};')
        
        with open(f"{parent_folder}/results.txt", 'w') as f:
            f.write(f'{vecSob_results_mapped}')
        return vecSob_results_mapped
if __name__ == "__main__":
    main()
