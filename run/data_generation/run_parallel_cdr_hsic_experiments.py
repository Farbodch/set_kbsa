#–----------------------------
# make this script visible to system
# and rest of the package visible
# to this script
#–----------------------------
from os import path as os_path
from sys import path as sys_path
script_dir = os_path.dirname(os_path.abspath(__file__))
scripts_bin_dir = os_path.dirname(script_dir)
project_root_dir = os_path.dirname(scripts_bin_dir)
sys_path.insert(0, project_root_dir)

#–----------------------------
# import dependencies
#–----------------------------
from auxiliary_utils.index_management import get_u_index_superset_onehot
from dolfin import MPI as dolfin_MPI
from auxiliary_utils.io_management import make_directory, write_to_textfile
from auxiliary_utils.mpi_management import get_per_rank_padded_indices, get_total_num_of_padding
from data_generation_scripts.cdr_hsic import cdr_hsic_experiment
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=3, required=True, help='Number of FEM simulations to generate.')
    parser.add_argument("--cdr_t_end", type=float, default=0.05, help='Final physical time (in seconds) of the CDR simulation.')
    parser.add_argument("--cdr_num_steps", type=int, default=500, help='Number of steps in the CDR simulations, in cdr_t_end/cdr_num_steps (s) increments. E.g., 500 steps for t_end of 0.05 (s) run at 0.1 (ms) per step.')
    user_inputs = parser.parse_args()
    N = user_inputs.N #number of total experiments to run
    cdr_num_steps = user_inputs.cdr_num_steps
    cdr_t_end = user_inputs.cdr_t_end

    # print("Beginning Experiment.")
    # t_start = timetime()
    comm = dolfin_MPI.comm_world
    # comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    padded_indices, local_num_of_padding = get_per_rank_padded_indices(N=N, size=size, rank=rank)

    cdr_params = {'mesh_directory': 'data/mesh_data/cdr/rectangle.xdmf', 
                't_end': cdr_t_end, #in seconds
                'num_steps': cdr_num_steps, #time steps, in t_end/num_steps increments, e.g., 100 steps for 0.01 t_end is 0.0001s, 0.1ms
                'return_bool': False,
                'mesh_steps': 0.025,
                'g_ineq_c': {'fuel': 0.02, 'oxygen': 0.14, 'product': 0.014, 'temp': 900}}
    
    #all workers need to have access to this
    u_indexSuperset_oneHot = get_u_index_superset_onehot(dim_of_U=5, higher_order=False)
    if rank == 0:
        total_num_of_padding_pre = get_total_num_of_padding(N=N, size=size)
        if total_num_of_padding_pre > 0:
            print(f"!----------!----------!\nPadding needed for MPI!\n>{total_num_of_padding_pre}< extra simulations will be run.\n!----------!----------!")

        parent_directory, parent_uid = make_directory(directory='data/experiment_data/cdr/hsic',
                                                    with_uid=True,
                                                    with_datetime=True, 
                                                    return_new_directory=True, 
                                                    return_uid=True)
    else:
        parent_directory = None
        parent_uid = None

    parent_directory = comm.bcast(parent_directory, root=0)
    parent_uid = comm.bcast(parent_uid, root=0) 

    local_results = [(i, cdr_hsic_experiment(u_indexSuperset_oneHot=u_indexSuperset_oneHot,
                            cdr_params=cdr_params,
                            mpi_rank=rank, 
                            parent_directory=parent_directory)) for i in padded_indices]
    
    total_num_of_padding_post = dolfin_MPI.sum(comm, local_num_of_padding)
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        try:
            assert total_num_of_padding_pre == total_num_of_padding_post
        except AssertionError:
            print(f'Assertion Warning! Predicted number of padded runs was {total_num_of_padding_pre}. Ran {total_num_of_padding_post} instead!')

        flat = [x for sub in all_results for x in sub]
        # print(all_results)
        print("Done:", len(flat), "tasks")

        if len(cdr_params['mesh_directory']) != 0:
            cdr_params.pop('mesh_steps')
        if not cdr_params['return_bool']:
            cdr_params.pop('g_ineq_c')
        
        content_to_write_to_txt_dict = {'parent_uid': parent_uid,
                                        'total_num_of_experiments': int(N+total_num_of_padding_post),
                                        'total_num_of_padded_runs': int(total_num_of_padding_post),
                                        'cdr_params': cdr_params}
        
        write_to_textfile(directory=parent_directory,
                        file_name='meta_data',
                        content_to_write_to_txt_dict=content_to_write_to_txt_dict)

    # print("Experiment Done.")
    # print(f"Execution time: {t_end - t_start:.6f} seconds")
    # parser = argparse.ArgumentParser(description="...")
    # parser.add_argument("--variable_name", default="default_value", help="tooltip-here")
    # args = parser.parse_args()
    # print(f"{args.variable_name}")
    comm.barrier()
    return 0

if __name__ == "__main__":
    main()