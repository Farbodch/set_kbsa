#–----------------------------
# make this script visible to system
# and rest of the package visible
# to this script
#–----------------------------
from os import path as os_path
from sys import path as sys_path
script_dir = os_path.dirname(os_path.abspath(__file__))
project_root_dir = os_path.dirname(script_dir)
sys_path.insert(0, project_root_dir)

#–----------------------------
# import dependencies
#–----------------------------
from numpy import (uint8,  # noqa: E402
                   array as np_array, 
                   zeros as np_zeros)
from hsic.hsic import hsic  # noqa: E402
from auxiliary_utils.io_management import (get_input_data_from_file_fenics_function, # noqa: E402
                                        load_mesh,
                                        load_function_space,
                                        write_from_dict_to_text_file)
from auxiliary_utils.mesh_management import (get_cell_markers_from_test_domain, # noqa: E402
                                             inject_cell_markers_from_numpy_array_to_MeshFunction,
                                             assemble_mass_matrix_from_mesh)
from auxiliary_utils.index_management import generator_order_r_idcs_as_onehot # noqa: E402
from hsic.hsic_utils import (sample_fenics_function, # noqa: E402
                             load_fenics_functions_as_indicator,
                             compute_integrated_gamma_matrix_mpi,
                             cleanup_checkpoint_files,
                             assemble_gamma_matrix_from_checkpoints)
from dolfin import (MPI as dolfin_MPI, parameters as df_parameters) # noqa: E402
import argparse # noqa: E402

import warnings # noqa: E402
warnings.filterwarnings("ignore", category=UserWarning)
df_parameters["reorder_dofs_serial"] = False
df_parameters["mesh_partitioner"] = "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100, required=True, help='Number of data points to use.')
    parser.add_argument("--M", type=int, default=100, required=False, help='Number of samplings per data point to estimate the set kernel.')
    parser.add_argument("--S", type=int, default=1, required=False, help='True for statistical estimation of the set kernel, False for explicitly integrating on FEM mesh.')
    user_inputs = parser.parse_args()
    N = user_inputs.N
    M = user_inputs.M
    S = user_inputs.S


    comm = dolfin_MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    process_model_names = ['2d_cdr', 'diffusion_1d']
    process_model_name = process_model_names[1]

    #2d cdr setting <- TO BE REFACTORED
    #1d diffusion setting <- TO BE REFACTORED
    P = 3
    valid_cdr_fields = ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']
    process_settings={'2d_cdr': {'data_directory': 'data/experiment_data/cdr/hsic',
                                'mesh_directory': 'data/mesh_data/cdr/rectangle.xdmf',
                                'u_domain_specifications': [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                                                            {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                                                            {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                                                            {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                                                            {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}],
                                'u_one_hot_key_map': {'10000': 'A', '01000': 'E', '00100': 'T_i', '00010': 'T_o', '00001': 'phi'},
                                'test_domain': np_array([[0.15, 0.5],[0.2, 0.3]]),
                                'g_constraint': 700,
                                'field_of_interest': valid_cdr_fields[3]},

                        'diffusion_1d': {'data_directory': 'data/experiment_data/diffusion_1d/hsic',
                                        'mesh_directory': 'data/mesh_data/diffusion_1d/h_128/interval_mesh.xdmf',
                                        'u_domain_specifications': [{'distribution_type': 'uniform', 'min': -1, 'max': 1} for _ in range(P)],
                                        'u_one_hot_key_map': {key: f'P_{P-i}' for i, key in enumerate(generator_order_r_idcs_as_onehot(r=1, d=P))},
                                        'test_domain': np_array([[0.45, 0.55]]),
                                        'P': P,
                                        'g_constraint': 0.135,
                                        'field_of_interest': 'diffusion_field'}}
    


    restart_toggle = False
    if S == 0:
        k_set_compute_statistical = False
    else:
        k_set_compute_statistical = True
    n = N
    m = M
    chunk_size_K_gamma_fem_explicit = 10
    shuffle_inputs = True
    return_hsic_results = False
    write_hsic_results_to_file = True
    '''
    cell_marker_policy (str): The rule for including a mesh cell, according to the bounding box defined by test_domain:
                      - 'all': Mark the cell if ALL of its vertices are inside
                               the test_domain. This creates an "inner bound" region.
                      - 'any': Mark the cell if AT LEAST ONE of its vertices
                               is inside the test_domain. This creates an "outer bound" region.
    '''
    cell_marker_policy = 'all'
    # data_dir_indices = None
    binary_system_output_data = None
    input_data_dirs_list_to_use = None
    field_data_dirs_list_to_use = None
    u_arr = None
    data_directory_with_uid = ''
    cell_markers = None
    cell_marker_numpy_array_extract = None
    #------------------------------------
    #Load in mesh data.
    #Note: load_mesh() loads the mesh into comm.comm_self for each individual worker.
    #------------------------------------
    my_mesh = load_mesh(mesh_dir=process_settings[process_model_name]['mesh_directory'])
    V = load_function_space(my_mesh)

    if rank == 0:

        #------------------------------------
        #load-in random-input data (u_arr)
        #------------------------------------
        num_of_u_inputs = len(process_settings[process_model_name]['u_domain_specifications'])
        u_arr, input_data_dirs_list_to_use = get_input_data_from_file_fenics_function(data_directory=process_settings[process_model_name]['data_directory'],
                                                                                num_of_u_inputs=num_of_u_inputs,
                                                                                n=n,
                                                                                shuffle=False,
                                                                                return_directories=True,
                                                                                adjust_for_mpi=True,
                                                                                mpi_size=size,
                                                                                process_type=process_model_name)
        
        n = len(u_arr)
        field_data_dirs_list_to_use = [dir.replace('input_data.npy', process_settings[process_model_name]['field_of_interest']) for dir in input_data_dirs_list_to_use]
        
        if not k_set_compute_statistical:
            cell_markers = get_cell_markers_from_test_domain(mesh=my_mesh,
                                                            test_domain=process_settings[process_model_name]['test_domain'],
                                                            cell_marker_policy=cell_marker_policy)
            cell_marker_numpy_array_extract = cell_markers.array()

        meta_data_dict = {'num_of_outer_loop_sampling_n': n,
                        'field_of_interest':process_settings[process_model_name]['field_of_interest'],
                        'test_domain':process_settings[process_model_name]['test_domain'],
                        'g_constraint':process_settings[process_model_name]['g_constraint']}
        if k_set_compute_statistical:
            meta_data_dict['k_set_compute_type'] = 'statistical'
            meta_data_dict['num_of_inner_loop_samplings_m'] = m
        else:
            meta_data_dict['k_set_compute_type'] = 'fem_explicit'
        k_set_compute_type = meta_data_dict['k_set_compute_type']
        data_directory_with_uid = write_from_dict_to_text_file(data_to_write_dict=meta_data_dict,
                                    data_file_name='meta_data',
                                    data_parent_directory=f'data/hsic_results/{process_model_name}/{k_set_compute_type}',
                                    generate_new_uid=True,
                                    return_data_directory_with_uid=True)
    #------------------------------------
    # Broadcast 
    #   data_directory_with_uid, 
    #   u_arr data, 
    #   cell_marker_numpy_array_extract, and 
    #   number of data points used (n) 
    # to all MPI workers, and schedule the static work distribution through 'indices'.
    #------------------------------------
    data_directory_with_uid = comm.bcast(data_directory_with_uid, root=0)
    u_arr = comm.bcast(u_arr, root=0)
    field_data_dirs_list_to_use = comm.bcast(field_data_dirs_list_to_use, root=0)
    if not k_set_compute_statistical:
        cell_marker_numpy_array_extract = comm.bcast(cell_marker_numpy_array_extract, root=0)
    n = comm.bcast(n, root=0)
    # max_jobs = (n + size - 1) // size 
    indices = range(rank, n, size)
    # num_of_padded_runs_in_curr_rank = 0
    # while len(indices) < max_jobs:
    #     num_of_padded_runs_in_curr_rank += 1
    # assert num_of_padded_runs_in_curr_rank==0
    comm.barrier()
    gamma_matrix = None
    if not k_set_compute_statistical:
        if rank != 0:
            cell_markers = inject_cell_markers_from_numpy_array_to_MeshFunction(mesh=my_mesh,
                                cell_marker_numpy_array_extract=cell_marker_numpy_array_extract)
        '''
        mass_matrix is built on-physical-node (individually per MPI worker)
        '''
        mass_matrix = assemble_mass_matrix_from_mesh(V=V, cell_markers=cell_markers)
        
    #------------------------------------
    #if k_set_compute_statistic is set to true, then the set kernel is 
    #computed via empirical uniform sampling using m samples.
    #------------------------------------
    if k_set_compute_statistical:
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
    
    #------------------------------------
    #if k_set_compute_statistic is set to false, then the set kernel is 
    #computed via directly integral definition of the symmetric difference metric 
    #on the FEM mesh nodes. 
    #
    #Assumption: FEM problem solved using using CG1 elements.
    #------------------------------------
    else:
        indicator_fens_list = load_fenics_functions_as_indicator(comm=comm,
                                                                V=V,
                                                                data_dirs_to_eval_list=field_data_dirs_list_to_use,
                                                                g_constraint=process_settings[process_model_name]['g_constraint'])
        comm.barrier()
        if restart_toggle:
            compute_integrated_gamma_matrix_mpi(comm=comm,
                                                mass_matrix=mass_matrix,
                                                indicator_fens_list=indicator_fens_list,
                                                chunk_size=chunk_size_K_gamma_fem_explicit,
                                                tmp_checkpoint_dir=f'{data_directory_with_uid}/checkpoints',
                                                restart=restart_toggle)
            gamma_matrix = assemble_gamma_matrix_from_checkpoints(comm=comm,
                                                                tmp_checkpoint_dir=f'{data_directory_with_uid}/checkpoints',
                                                                n=n)
        else:
            gamma_matrix = compute_integrated_gamma_matrix_mpi(comm=comm,
                                                            mass_matrix=mass_matrix,
                                                            indicator_fens_list=indicator_fens_list,
                                                            chunk_size=chunk_size_K_gamma_fem_explicit,
                                                            tmp_checkpoint_dir=f'{data_directory_with_uid}/checkpoints',
                                                            restart=restart_toggle)
        comm.barrier()
        # print(field_data_dirs_list_to_use)
        # print(gamma_matrix)
    if rank==0:
        hsic_results = hsic(u_domain_specifications=process_settings[process_model_name]['u_domain_specifications'],
                        u_one_hot_key_map=process_settings[process_model_name]['u_one_hot_key_map'],
                        binary_system_output_data=binary_system_output_data,
                        u_arr=u_arr,
                        vectorized_hsic_flag=True,
                        vectorized_K_U_flag=True,
                        gamma_matrix=gamma_matrix)
        if write_hsic_results_to_file:
            with open(f"{data_directory_with_uid}/results.txt", 'w') as f:
                f.write(f'{hsic_results}')
        if not k_set_compute_statistical:
            cleanup_checkpoint_files(comm=comm, tmp_checkpoint_dir=f'{data_directory_with_uid}/checkpoints')
        if return_hsic_results:
            return hsic_results

if __name__ == "__main__":
    main()
