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
                #    arange as np_arange)
# from numpy.random import shuffle as np_shuffle
from vecSob.vecSob import vecSob_statistical_parallelized, integrated_spatial_generalizedSob_parallelized # noqa: E402
from auxiliary_utils.io_management import (get_data_file_directories, # noqa: E402
                                        load_mesh,
                                        load_function_space)
from auxiliary_utils.index_management import generator_order_r_idcs_as_onehot # noqa: E402
from auxiliary_utils.mesh_management import (get_cell_markers_from_test_domain, # noqa: E402
                                             inject_cell_markers_from_numpy_array_to_MeshFunction,
                                             assemble_mass_matrix_from_mesh)
from vecSob.vecSob_utils import (sample_fenics_function, # noqa: E402
                                 build_sorted_output_data_dict) 
# from auxiliary_utils.mpi_management import adjust_sampling_number_for_hsic
from dolfin import MPI as dolfin_MPI # noqa: E402
from datetime import datetime # noqa: E402
from time import time as timetime # noqa: E402
from uuid import uuid4 # noqa: E402
from os import makedirs # noqa: E402
import gc # noqa: E402
from random import shuffle as rnd_shuffle # noqa: E402
import argparse # noqa: E402


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100, required=True, help='Number of data points to use.')
    #UPDATE help TEXT
    parser.add_argument("--H", type=int, default=100, required=False, help='Number of samplings per data point to estimate the set kernel.')
    #UPDATE help TEXT
    parser.add_argument("--S", type=int, default=1, required=False, help='True for statistical estimation of the set kernel, False for explicitly integrating on FEM mesh.')
    #UPDATE help TEXT
    parser.add_argument("--M", type=int, default=1024, required=False, help='True for statistical estimation of the set kernel, False for explicitly integrating on FEM mesh.')
    #UPDATE help TEXT
    parser.add_argument("--shuffle", type=int, default=0, required=False, help='True for statistical estimation of the set kernel, False for explicitly integrating on FEM mesh.')
    user_inputs = parser.parse_args()
    N = user_inputs.N
    h = user_inputs.H
    S = user_inputs.S
    m = user_inputs.M
    shuffle = user_inputs.shuffle
    if S == 1:
        compute_explicit_FEM = False
        num_of_grid_points = h
    else:
        compute_explicit_FEM = True
    num_of_mesh_steps = m

    n = N
    comm = dolfin_MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    process_model_names = ['2d_cdr', 'diffusion_1d']
    process_model_name = process_model_names[1]
    
    if rank == 0:
        print(6*'-',f'Starting run on model: {process_model_name} | with n = {n} | mesh_steps = {num_of_mesh_steps} | explicit_FEM = {compute_explicit_FEM}',6*'-')

    P=3
    valid_cdr_fields = ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']
    process_settings={'2d_cdr': {'data_directory': 'data/experiment_data/cdr/vecSob',
                                'mesh_directory': 'data/mesh_data/cdr/rectangle.xdmf',
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
                                'get_total_sobols': True,
                                'process_type': 'cdr_vecSob'},
    
                        'diffusion_1d': {'data_directory': '/Users/farbodchamanian/Desktop/P--j-ct/code_repository/set_kbsa/data/experiment_data/diffusion_1d/vecSob',
                                        # 'data_directory': './data/experiment_data/diffusion_1d/vecSob',
                                        'mesh_directory': f'/Users/farbodchamanian/Desktop/P--j-ct/code_repository/set_kbsa/data/mesh_data/diffusion_1d/h_{num_of_mesh_steps}/interval_mesh.xdmf',
                                        'u_domain_specifications': [{'distribution_type': 'uniform', 'min': -1, 'max': 1} for _ in range(P)],
                                        'u_one_hot_key_map': {key: f'P_{P-i}' for i, key in enumerate(generator_order_r_idcs_as_onehot(r=1, d=P))},
                                        'test_domain': np_array([[0.4, 0.6]]),
                                        'P': P,
                                        'g_constraint': 0.135,
                                        'field_of_interest': 'diffusion_field',
                                        'which_orders': [1],
                                        'get_total_sobols': True,
                                        'process_type': 'diffusion_1d_vecSob'}}
    
    
    #should be power of the num of dimensions
    if shuffle == 0:
        shuffle_inputs = False
    else:
        shuffle_inputs = True

    '''
    cell_marker_policy (str): The rule for including a mesh cell, according to the bounding box defined by test_domain:
                      - 'all': Mark the cell if ALL of its vertices are inside
                               the test_domain. This creates an "inner bound" region.
                      - 'any': Mark the cell if AT LEAST ONE of its vertices
                               is inside the test_domain. This creates an "outer bound" region.
    '''
    cell_marker_policy = 'all'

    binary_system_output_data = None
    data_file_dirs_dict = None

    """
    binary_system_output_data = {index_str: binary_system_output_data_A}
    binary_system_output_data[key].shape needs to be (3, n, h) where
    [0,:,:] contains y_I, [1,:,:] contains y_II and [2,:,:] y_tilde. 
    """
    my_mesh = load_mesh(mesh_dir=process_settings[process_model_name]['mesh_directory'])
    V = load_function_space(my_mesh)
    cell_markers = None
    cell_marker_numpy_array_extract = None
    if rank == 0:
        if compute_explicit_FEM:
            cell_markers = get_cell_markers_from_test_domain(mesh=my_mesh,
                                                            test_domain=process_settings[process_model_name]['test_domain'],
                                                            cell_marker_policy=cell_marker_policy)
            cell_marker_numpy_array_extract = cell_markers.array()
        num_of_u_inputs = len(process_settings[process_model_name]['u_domain_specifications'])
        
        #get all relevant directories
        data_files_dirs_list_bag = get_data_file_directories(base_dir=process_settings[process_model_name]['data_directory'], 
                                                    data_type='input_data',
                                                    process_type=process_settings[process_model_name]['process_type'],
                                                    verbose=False,
                                                    enforce_params=False,
                                                    num_of_mesh_steps=num_of_mesh_steps)
        #adjust target file names to the fenics function field of interest
        field_data_dirs_list_bag = [dir.replace('input_data.npy', process_settings[process_model_name]['field_of_interest']) for dir in data_files_dirs_list_bag]
        # if shuffle_inputs:
        #     rnd_shuffle(field_data_dirs_list_bag)
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

    data_file_dirs_dict = comm.bcast(data_file_dirs_dict, root=0)
    if compute_explicit_FEM:
        cell_marker_numpy_array_extract = comm.bcast(cell_marker_numpy_array_extract, root=0)
    n = comm.bcast(n, root=0)
    indices = range(rank, n, size)
    comm.barrier()

    if compute_explicit_FEM:
        if rank != 0:
            cell_markers = inject_cell_markers_from_numpy_array_to_MeshFunction(mesh=my_mesh,
                                cell_marker_numpy_array_extract=cell_marker_numpy_array_extract)
        '''
        mass_matrix is built on-physical-node (individually per MPI worker)
        '''
        mass_matrix = assemble_mass_matrix_from_mesh(V=V, cell_markers=cell_markers)

        vecSob_results_mapped = integrated_spatial_generalizedSob_parallelized(comm=comm,
                                                                               mass_matrix=mass_matrix,
                                                                               V=V,
                                                                               g_constraint=process_settings[process_model_name]['g_constraint'],
                                                                               data_file_dirs_dict=data_file_dirs_dict,
                                                                               u_one_hot_key_map=process_settings[process_model_name]['u_one_hot_key_map'],
                                                                               which_orders=process_settings[process_model_name]['which_orders'],
                                                                               num_of_u_inputs=len(process_settings[process_model_name]['u_domain_specifications']),
                                                                               get_total_sobols=process_settings[process_model_name]['get_total_sobols'])


    else:
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
        vecSob_results_mapped = vecSob_statistical_parallelized(comm=comm,
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
                f.write(f'parent_uid_{parent_uid};\nnum_of_outer_loop_sampling_n:{n};\nfield_of_interest:{field_of_interest};\ntest_domain:{test_domain};\ng_constraint:{g_constraint};\n')
                if compute_explicit_FEM:
                    f.write('compute_method:explicit_FEM;')
                else:
                    f.write('compute_method:statistical;')
                    f.write(f'\nnum_of_grid_points:{num_of_grid_points};')
                f.write(f'\nmesh_num_of_steps:{num_of_mesh_steps};')
        vecSob_results_mapped_string = f'{vecSob_results_mapped}'.replace('),', ',').replace('np.float64(','').replace(')}','}')
        with open(f"{parent_folder}/results.txt", 'w') as f:
            f.write(vecSob_results_mapped_string)
        # return vecSob_results_mapped 
if __name__ == "__main__":
    main()
