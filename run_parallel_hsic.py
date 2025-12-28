from numpy import (uint8, 
                   array as np_array,
                   zeros as np_zeros)
                #    arange as np_arange)
# from numpy.random import shuffle as np_shuffle
from hsic.hsic import hsic
from auxiliary_utils.io_management import (get_input_data_from_file_fenics_function,
                                        load_mesh,
                                        load_function_space)
from hsic.hsic_utils import sample_fenics_function
# from auxiliary_utils.mpi_management import adjust_sampling_number_for_hsic
from dolfin import MPI as dolfin_MPI
from datetime import datetime
from time import time as timetime
from uuid import uuid4
from os import makedirs

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    comm = dolfin_MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    valid_cdr_fields = ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']
    process_model_name = '2d_cdr'
    #DATA_DIRECTORY NOT UNIFIED!! FIX THIS! (LRZ vs Home_dir locations are different.)
    process_settings={'2d_cdr': {'data_directory': 'data/experiment_data',
                                'fem_mesh_directory': 'data/CDR/mesh_save_dir/rectangle.xdmf',
                                'u_domain_specifications': [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                                                            {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                                                            {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                                                            {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                                                            {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}],
                                'u_one_hot_key_map': {'10000': 'A', '01000': 'E', '00100': 'T_i', '00010': 'T_o', '00001': 'phi'},
                                'test_domain': np_array([[0.5, 0.15],[0.2, 0.3]]),
                                'g_constraint': 700,
                                'field_of_interest': valid_cdr_fields[3]}}
    
    n = 10000
    m = 20000
    shuffle_inputs = True

    # if size > 1:
    parallelize_flag = True
    # else:
        # parallelize_flag = False
    # data_dir_indices = None
    binary_system_output_data = None
    input_data_dirs_list_to_use = None
    field_data_dirs_list_to_use = None
    u_arr = None
    
    if parallelize_flag:
        if rank == 0:
            num_of_u_inputs = len(process_settings[process_model_name]['u_domain_specifications'])
            u_arr, input_data_dirs_list_to_use = get_input_data_from_file_fenics_function(data_directory=process_settings[process_model_name]['data_directory'],
                                                                                   num_of_u_inputs=num_of_u_inputs,
                                                                                   n=n,
                                                                                   shuffle=True,
                                                                                   return_directories=True,
                                                                                   adjust_for_mpi=True,
                                                                                   mpi_size=size)
           
            n = len(u_arr)
            field_data_dirs_list_to_use = [dir.replace('input_data.npy', process_settings[process_model_name]['field_of_interest']) for dir in input_data_dirs_list_to_use]
        u_arr = comm.bcast(u_arr, root=0)
        field_data_dirs_list_to_use = comm.bcast(field_data_dirs_list_to_use, root=0)
        n = comm.bcast(n, root=0)
        max_jobs = (n + size - 1) // size 
        indices = range(rank, n, size)
        num_of_padded_runs_in_curr_rank = 0
        while len(indices) < max_jobs:
            num_of_padded_runs_in_curr_rank += 1
        assert num_of_padded_runs_in_curr_rank==0
        
        my_mesh = load_mesh(mesh_dir=process_settings[process_model_name]['fem_mesh_directory'])
        V = load_function_space(my_mesh)

        binary_system_output_data_local = np_zeros((n, m), dtype=uint8)

        for i in indices:
            binary_system_output_data_local[i] = sample_fenics_function(data_directory=field_data_dirs_list_to_use[i],
                                                                        mesh=my_mesh,
                                                                        func_space_V=V,
                                                                        test_domain=process_settings[process_model_name]['test_domain'],
                                                                        num_of_spatial_sampling_m=m,
                                                                        g_constraint=process_settings[process_model_name]['g_constraint'])
        local_results = [(i, binary_system_output_data_local[i].copy()) for i in indices]
        all_results = comm.gather(local_results, root=0)
        comm.barrier()
        if rank==0:
            binary_system_output_data = np_zeros((n, m), dtype=uint8)
            for rank_list in all_results:
                for i, results in rank_list:
                    binary_system_output_data[i] = results
        comm.barrier()
    if rank==0:
        hsic_results = hsic(u_domain_specifications=process_settings[process_model_name]['u_domain_specifications'],
                        u_one_hot_key_map=process_settings[process_model_name]['u_one_hot_key_map'],
                        binary_system_output_data=binary_system_output_data,
                        u_arr=u_arr,
                        vectorized_hsic_flag=True,
                        vectorized_K_U_flag=True)
        print(hsic_results)
        parent_uid = datetime.now().strftime("%Y_%m_%d_%H_%M_%S__") + str(uuid4().hex)
        parent_folder = f"data/hsic_results/{parent_uid}"
        makedirs(parent_folder, exist_ok=True)
        with open(f"{parent_folder}/meta_data.txt", 'w') as f:
                field_of_interest = process_settings[process_model_name]['field_of_interest']
                test_domain = process_settings[process_model_name]['test_domain']
                g_constraint = process_settings[process_model_name]['g_constraint']
                f.write(f'parent_uid_{parent_uid};\nnum_of_outer_loop_sampling_n:{n};\nnum_of_inner_loop_samplings_m{m};\nfield_of_interest:{field_of_interest};\ntest_domain:{test_domain};\ng_constraint:{g_constraint};\ndata_shuffle:{shuffle_inputs};')
        with open(f"{parent_folder}/results.txt", 'w') as f:
            f.write(f'{hsic_results}')
        return hsic_results

if __name__ == "__main__":
    main()
