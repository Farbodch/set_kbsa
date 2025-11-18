from mpi4py import MPI
from datetime import datetime
import time
import uuid
from os import makedirs
from kbsa.utils_kbsa import _get_u_indexSuperset_oneHot
from kbsa.model_generators import get_CDR
from utils.math_utils import gen_uniform, gen_log_uniform
from numpy import array as np_arr
from numpy import save as np_save
from dolfin import XDMFFile, Mesh

def work(u_indexSuperset_oneHot, cdr_params, rank, parent_path):
    """
    we're sampling the random-inputs here (save to u_sampled), 
    then define a generator by calling f=get_CDR(), 
    then run results_fens_list = f(u=u_sampled).
    results_fens_list contains 4 'dolfin.function.function.Function' objects in it (list),
    where each object represents an observable field, where one can query by 
    submitting spatial coordinates to (eg results_fens_list[0](0.1,0.2)). 
    work() then needs to write every one of the 4 functions to an XDMFFile encoded 
    with an accompanying HDF5 formatted data file.
    work() will also need to write u_sampled to an accompanying .txt file.
    IMPORTANT: the functions data WITHOUT the .txt file info is USELESS for SA!
    """
    local_uid = str(uuid.uuid4().hex)
    local_path = f"{parent_path}/{local_uid}"
    makedirs(local_path, exist_ok=True)

    u_A = gen_log_uniform(a=5.5e11, b=1.5e12, size=6)
    u_E = gen_log_uniform(a=1.5e3, b=9.5e3, size=6)
    u_T_i = gen_uniform(a=200, b=400, size=6)
    u_T_o = gen_uniform(a=850, b=1000, size=6)
    u_phi = gen_uniform(a=0.5, b=1.5, size=6)
    u_input_all_A = np_arr([u_A, u_E, u_T_i, u_T_o, u_phi])
    
    mesh_path = cdr_params["mesh_2D_dir"]
    
    
    for idx_A, idx_A_str in enumerate(u_indexSuperset_oneHot):
        # makedirs(parent_path, exist_ok=True)
        local_path_idx_A = f"{local_path}/{idx_A_str}"
        curr_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        makedirs(local_path_idx_A, exist_ok=True)
        
        u_input_idx_A = u_input_all_A[:, idx_A]
        simul_t0 = time.time()
        cdr_fen = get_CDR(params=cdr_params)
        results_list = cdr_fen(u=u_input_idx_A)
        simul_t1 = time.time()
        simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"
        
        #store results to file
        #-----------------
        #store input data
        np_save(file=f"{local_path_idx_A}/input_data_{idx_A_str}.npy", arr=u_input_idx_A)
        #store output data
        mesh = Mesh()
        with XDMFFile(f"{mesh_path}") as xdmf:
            xdmf.read(mesh)
        with XDMFFile(mesh.mpi_comm(), f'{local_path_idx_A}/fuel_field') as xdmf_1:
            xdmf_1.write_checkpoint(results_list[0], 'fuel_field', 0, XDMFFile.Encoding.HDF5, append=False)
        with XDMFFile(mesh.mpi_comm(), f'{local_path_idx_A}/oxygen_field') as xdmf_2:
            xdmf_2.write_checkpoint(results_list[1], 'oxygen_field', 0, XDMFFile.Encoding.HDF5, append=False)
        with XDMFFile(mesh.mpi_comm(), f'{local_path_idx_A}/product_field') as xdmf_3:
            xdmf_3.write_checkpoint(results_list[2], 'product_field', 0, XDMFFile.Encoding.HDF5, append=False)   
        with XDMFFile(mesh.mpi_comm(), f'{local_path_idx_A}/temp_field') as xdmf_4:
            xdmf_4.write_checkpoint(results_list[3], 'temp_field', 0, XDMFFile.Encoding.HDF5, append=False)
        #store local_uid, core rank, index of input_A, u_input data (redundancy), simulation time of this 1 iteration
        with open(f"{local_path_idx_A}/meta_data.txt", 'w') as f:
            f.write(f'local_uid_{local_uid};\ndatetime_{curr_datetime};\nrank_{rank};\n{idx_A_str};\ninput_data:{u_input_idx_A};\n{simul_time_str};')
    return 'done.'


def main():
    print("Beginning Experiment.")
    t_start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 10 #number of total experiments to run
    cdr_params = {'mesh_2D_dir': 'data/CDR/mesh_save_dir/rectangle.xdmf', 
                        't_end': 0.05, #in seconds
                        'num_steps': 500, #time steps are 0.0001s, 0.1ms
                        'return_bool': False}
    
    #all workers need to have access to this
    u_indexSuperset_oneHot = _get_u_indexSuperset_oneHot(dim_of_U=5, higher_order=False)
    if rank == 0:
        parent_uid = datetime.now().strftime("%Y_%m_%d_%H_%M_%S__uid_") + str(uuid.uuid4().hex)
        parent_folder = f"experimental_data/{parent_uid}"
        makedirs(parent_folder, exist_ok=True)
    else:
        parent_uid = None 
    parent_uid = comm.bcast(parent_uid, root=0)
    parent_folder = f"experimental_data/{parent_uid}"

    indices = range(rank, N, size)
    
    local_results = [(i, work(u_indexSuperset_oneHot=u_indexSuperset_oneHot,
                            cdr_params=cdr_params,
                            rank=rank, 
                            parent_path=parent_folder),
                        parent_folder) for i in indices]
    
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat = [x for sub in all_results for x in sub]
        print(all_results)
        print("Done:", len(flat), "tasks")
    t_end = time.time()
    with open(f"{parent_folder}/meta_data.txt", 'w') as f:
            f.write(f'parent_uid_{parent_uid};\n\nexperiment_duration_{t_end - t_start:.6f};\ncdr_params_{cdr_params};')
    
    print("Experiment Done.")
    print(f"Execution time: {t_end - t_start:.6f} seconds")
    # parser = argparse.ArgumentParser(description="...")
    # parser.add_argument("--variable_name", default="default_value", help="tooltip-here")
    # args = parser.parse_args()
    # print(f"{args.variable_name}")
    return 0

if __name__ == "__main__":
    main()