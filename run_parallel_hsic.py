from numpy import ndarray, uint8, array as np_array
from numpy import zeros as np_zeros
from numpy import arange as np_arange
from numpy.random import uniform as np_unif
from numpy.random import shuffle as np_shuffle
from kbsa.hsic import hsic
from kbsa.hsic_utils import get_data_file_dirs
from dolfin import Mesh, FunctionSpace, Function, XDMFFile, MPI as dolfin_MPI
from pathlib import Path
from datetime import datetime
from time import time as timetime
from uuid import uuid4
from os import makedirs
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_mesh_worker(mesh_dir: str = "data/CDR/mesh_save_dir/rectangle.xdmf"):
    fenics_comm = dolfin_MPI.comm_self
    mesh = Mesh(fenics_comm)
    with XDMFFile(fenics_comm, mesh_dir) as xdmf:
        xdmf.read(mesh)
    return mesh
def load_function_space_worker(mesh: Mesh, cg_order=1):
    V = FunctionSpace(mesh, 'CG', cg_order) 
    return V

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

def main():

    comm = dolfin_MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()

    mesh_directory = 'data/CDR/mesh_save_dir/rectangle.xdmf'
    valid_cdr_fields = ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']
    field_of_interest = valid_cdr_fields[1]
    data_directory = "data/experiment_data"
    test_domain = np_array([[0.5, 0.15],[0.2, 0.3]])
    u_domain_specs = [{'distribution_type': 'log_uniform', 'min': 5.5e11, 'max': 1.5e12},
                    {'distribution_type': 'log_uniform', 'min': 1.5e3, 'max': 9.5e3},
                    {'distribution_type': 'uniform', 'min': 850, 'max': 1000},
                    {'distribution_type': 'uniform', 'min': 200, 'max': 400},
                    {'distribution_type': 'uniform', 'min': 0.5, 'max': 1.5}]
    key_map_2d_cdr = {'10000': 'A', 
                    '01000': 'E',
                    '00100': 'T_i',
                    '00010': 'T_o',
                    '00001': 'phi'}
    g_constraint = 700
    n = 10000
    m = 20000
    shuffle_inputs = True

    if size > 1:
        parallelize_flag = True
    else:
        parallelize_flag = False
    data_dir_indices = None
    binary_system_output_data = None
    input_data_dirs_list_to_use = None
    field_data_dirs_list_to_use = None
    if parallelize_flag:
        if rank == 0:
            n_padded_up = n + (size-n%size)
            n_padded_down = n - (n%size)
            input_data_dirs_list = get_data_file_dirs(data_directory, data_type='input_data')
            
            n_max = len(input_data_dirs_list)
            data_dir_indices = np_arange(0, n_max)
            if shuffle_inputs:
                np_shuffle(data_dir_indices)

            if n_padded_up > n_max:
                print(f'Using {n-n_padded_down} less data points to ensure MPI parallelization. Need {n_padded_up} data points at the next parallelization increment using {size} MPI tasks/core.')
                n = n_padded_down
            else:
                print(f'Using {n_padded_up-n} extra data points to ensure MPI parallelization.')
                n = n_padded_up
            
            data_dir_indices = data_dir_indices[:n]
            input_data_dirs_list_to_use = [input_data_dirs_list[i] for i in data_dir_indices[:n]]
            field_data_dirs_list_to_use = [dir.replace('input_data.npy', field_of_interest) for dir in input_data_dirs_list]

        input_data_dirs_list_to_use = comm.bcast(input_data_dirs_list_to_use, root=0)
        field_data_dirs_list_to_use = comm.bcast(field_data_dirs_list_to_use, root=0)
        data_dir_indices = comm.bcast(data_dir_indices, root=0)
        n = len(data_dir_indices)
        
        max_jobs = (n + size - 1) // size 
        indices = range(rank, n, size)
        num_of_padded_runs_in_curr_rank = 0
        while len(indices) < max_jobs:
            # indices.append(None)
            num_of_padded_runs_in_curr_rank += 1
        assert num_of_padded_runs_in_curr_rank==0
        
        my_mesh = load_mesh_worker(mesh_dir=mesh_directory)
        V = load_function_space_worker(my_mesh)

        binary_system_output_data_local = np_zeros((n, m), dtype=uint8)

        for i in indices:
            dir = field_data_dirs_list_to_use[i]
            binary_system_output_data_local[i] = sample_fenics_function(field_file_path=dir,
                                                                                mesh=my_mesh,
                                                                                func_space_V=V,
                                                                                test_domain=test_domain,
                                                                                num_of_spatial_sampling_m=m,
                                                                                g_constraint=g_constraint)
        

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
        hsic_results = hsic(data_directory=data_directory,
                            field_of_interest=field_of_interest,
                            n=n,
                            num_of_spatial_sampling_m=m,
                            test_domain=test_domain,
                            u_domain_specs=u_domain_specs,
                            g_constraint=g_constraint,
                            shuffle_inputs=shuffle_inputs,
                            u_one_hot_key_map=key_map_2d_cdr,
                            binary_system_output_data=binary_system_output_data,
                            input_data_dirs_to_use_parall_processed=input_data_dirs_list_to_use)
        print(hsic_results)
        parent_uid = datetime.now().strftime("%Y_%m_%d_%H_%M_%S__") + str(uuid4().hex)
        parent_folder = f"data/hsic_results/{parent_uid}"
        makedirs(parent_folder, exist_ok=True)
        with open(f"{parent_folder}/meta_data.txt", 'w') as f:
                f.write(f'parent_uid_{parent_uid};\nnum_of_outer_loop_sampling_n:{n};\nnum_of_inner_loop_samplings_m{m};\nfield_of_interest:{field_of_interest};\ntest_domain:{test_domain};\ng_constraint:{g_constraint};\ndata_shuffle:{shuffle_inputs};')
        with open(f"{parent_folder}/results.txt", 'w') as f:
            f.write(f'{hsic_results}')
        return hsic_results

if __name__ == "__main__":
    main()
