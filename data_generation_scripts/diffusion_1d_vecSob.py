from time import time as timetime
from numeric_models.pde_models import get_1D_diff_FEM
from numeric_models.numeric_models_utils import generate_data
from numpy import (int8, array as np_arr, 
                   save as np_save,
                   where as np_where)
from dolfin import Mesh, XDMFFile, MPI as dolfin_MPI
from auxiliary_utils.io_management import make_directory, write_to_textfile


def _store_output_data(fenics_comm, results, local_path_idx_A, mesh_directory):
    # store output data
    mesh = Mesh(fenics_comm)
    with XDMFFile(fenics_comm, mesh_directory) as xdmf:
        xdmf.read(mesh)
    with XDMFFile(fenics_comm, f'{local_path_idx_A}/diffusion_field') as xdmf:
            xdmf.write_checkpoint(results, 'diffusion_field', 0, XDMFFile.Encoding.HDF5, append=False)

def _run_diffusion_1d(fenics_comm, 
             params, 
             u, 
             idx_A_str, 
             mpi_rank,
             local_uid,
             verbose: bool = True, 
             return_sim_time: bool = True):

    simul_t0 = timetime()
    if verbose:
        print(f'Started work. Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
    diffusion_1d_fen = get_1D_diff_FEM(params=params, comm=fenics_comm, local_uid=local_uid)
    results = diffusion_1d_fen(u=u)

    if verbose:
        print(f'Work done! Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
    simul_t1 = timetime()
    simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"
    if return_sim_time:
        return results, simul_time_str
    else:
        return results

def _run_experiment(params, u_input_idx_A, mesh_directory, fenics_comm, mpi_rank, local_path_idx_A, local_uid, idx_A_str):
    make_directory(directory=local_path_idx_A,
                with_uid=False,
                with_datetime=False,
                return_new_directory=False,
                return_uid=False)

    results, simul_time_str = _run_diffusion_1d(fenics_comm=fenics_comm, 
                                                params=params, 
                                                u=u_input_idx_A,
                                                idx_A_str=idx_A_str,
                                                mpi_rank=mpi_rank, 
                                                local_uid=local_uid, 
                                                verbose=True, 
                                                return_sim_time=True)
    #-----------------
    # store results to file
    #-----------------
    # store input data
    np_save(file=f"{local_path_idx_A}/input_data.npy", arr=u_input_idx_A)
    # store output data
    _store_output_data(fenics_comm, results, local_path_idx_A, mesh_directory)
    # store local_uid, core rank, index of input_A, u_input data (redundancy), simulation time of this 1 iteration
    content_to_write_to_txt_dict = {'local_uid': local_uid,
                                    'rank': mpi_rank,
                                    'idx_A': idx_A_str,
                                    'input_data': u_input_idx_A,
                                    'simulation_time': simul_time_str}
    write_to_textfile(directory=local_path_idx_A, 
                    file_name='meta_data',
                    content_to_write_to_txt_dict=content_to_write_to_txt_dict,
                    include_current_datetime=True)

def diffusion_1d_vecSob_experiment(index_set_to_calculate, 
                        params, mpi_rank, 
                        parent_directory, 
                        make_directory_with_uid: bool=True,
                        make_directory_with_datetime: bool=False):
    """
    Sampling random-inputs 3 times -> this results in u_1, u_2, u_3 -> save to dir_main
    run model for u_1 and u_3 -> save to dir_main
    depending on order numbers given -> for each order r (from 1 to u_dim-1):
        - for each A\in\powerset(u_dim) s.t. |A|=r:
            - set u_tilde = (u_{1,A}, u_{2,A^c})
            - create a folder named str(A) in dir_main
            - run model for u_tilde
            - store in dir_main/str(A)


    we're sampling the random-inputs here (save to u_sampled), 
    then define a generator by calling f=get_1D_diff_FEM(), 
    then run results_fens_list = f(u=u_sampled).
    results_fens_list contains 4 'dolfin.function.function.Function' objects in it (list),
    where each object represents an observable field, where one can query by 
    submitting spatial coordinates to (eg results_fens_list[0](0.1,0.2)). 
    work() then needs to write every one of the 4 functions to an XDMFFile encoded 
    with an accompanying HDF5 formatted data file.
    work() will also need to write u_sampled to an accompanying .txt file.
    IMPORTANT: the functions data without the .txt file info is useless for SA!
    """
    fenics_comm = dolfin_MPI.comm_self

    local_directory, local_uid = make_directory(directory=parent_directory,
                                                with_uid=make_directory_with_uid,
                                                with_datetime=make_directory_with_datetime, 
                                                return_new_directory=True, 
                                                return_uid=True)
    depth_P = params["P"]
    # choosing range(depth_P) for individual indices for each randomized-input 
    # (P_1 ...01, P_2 ...10, P_3 ...100, etc.) 
    # and size=3 for the three input-data-sets required for the pick-freeze algorithm
    # u_P = generate_data('uniform', min_u=-1, max_u=1, size=3)
    u_all_realizations = np_arr([generate_data('uniform', min_u=-1, max_u=1, size=3) for _ in range(depth_P)])
    u_I = u_all_realizations[:, 0]
    u_II = u_all_realizations[:, 1]
    u_III = u_all_realizations[:, 2]
    mesh_directory = params["mesh_directory"]
    _run_experiment(params = params, 
                    u_input_idx_A=u_I, 
                    mesh_directory=mesh_directory, 
                    fenics_comm=fenics_comm, 
                    mpi_rank=mpi_rank, 
                    local_path_idx_A=f"{local_directory}/u_I", 
                    local_uid=local_uid, 
                    idx_A_str='I')
    _run_experiment(params = params, 
                    u_input_idx_A=u_II, 
                    mesh_directory=mesh_directory, 
                    fenics_comm=fenics_comm, 
                    mpi_rank=mpi_rank, 
                    local_path_idx_A=f"{local_directory}/u_II", 
                    local_uid=local_uid, 
                    idx_A_str='II')
    u_III_directory = f"{local_directory}/u_III"
    make_directory(directory=u_III_directory,
                with_uid=False,
                with_datetime=False,
                return_new_directory=False,
                return_uid=False)
    np_save(file=f"{u_III_directory}/input_data.npy", arr=u_III)
    content_to_write_to_txt_dict = {'local_uid': local_uid,
                                    'rank': mpi_rank,
                                    'idx_A': "III",
                                    'input_data': u_III,
                                    'simulation_time': '0.0'}
    write_to_textfile(directory=u_III_directory, 
                    file_name='meta_data',
                    content_to_write_to_txt_dict=content_to_write_to_txt_dict,
                    include_current_datetime=True)
    """
    NOW -> Based on maybe an index_set_to_calculate -> combine different A from U_[0] and A^C from U_[2] -> run and save
    To Do -> Just need to implement a combinating-mechanism based on A \in index_set_to_calculate as we loop through index_set_to_calculate
    """
    for _, idx_A_str in enumerate(index_set_to_calculate):
        u_tilde = np_where(np_arr(list(idx_A_str), dtype=int8) == 1, u_II, u_III)
        _run_experiment(params = params, 
                    u_input_idx_A=u_tilde, 
                    mesh_directory=mesh_directory, 
                    fenics_comm=fenics_comm, 
                    mpi_rank=mpi_rank, 
                    local_path_idx_A=f"{local_directory}/{idx_A_str}", 
                    local_uid=local_uid, 
                    idx_A_str=idx_A_str)