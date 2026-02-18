from time import time as timetime
from numeric_models.pde_models import get_CDR
from numeric_models.numeric_models_utils import generate_data
from numpy import (int8, array as np_arr, save as np_save, where as np_where)
from dolfin import XDMFFile, Mesh, MPI as dolfin_MPI
from auxiliary_utils.io_management import make_directory, write_to_textfile

def _store_output_data(fenics_comm, results_list, local_path_idx_A, mesh_path):
    #store output data
    mesh = Mesh(fenics_comm)
    with XDMFFile(fenics_comm, mesh_path) as xdmf:
        xdmf.read(mesh)
    with XDMFFile(fenics_comm, f'{local_path_idx_A}/fuel_field') as xdmf_1:
        xdmf_1.write_checkpoint(results_list[0], 'fuel_field', 0, XDMFFile.Encoding.HDF5, append=False)
    with XDMFFile(fenics_comm, f'{local_path_idx_A}/oxygen_field') as xdmf_2:
        xdmf_2.write_checkpoint(results_list[1], 'oxygen_field', 0, XDMFFile.Encoding.HDF5, append=False)
    with XDMFFile(fenics_comm, f'{local_path_idx_A}/product_field') as xdmf_3:
        xdmf_3.write_checkpoint(results_list[2], 'product_field', 0, XDMFFile.Encoding.HDF5, append=False)   
    with XDMFFile(fenics_comm, f'{local_path_idx_A}/temp_field') as xdmf_4:
        xdmf_4.write_checkpoint(results_list[3], 'temp_field', 0, XDMFFile.Encoding.HDF5, append=False)

def _run_cdr(fenics_comm, cdr_params, u, idx_A_str, mpi_rank, local_uid, verbose: bool=True, return_sim_time: bool=True):
    simul_t0 = timetime()
    if verbose:
        print(f'Started work. Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
    cdr_fen = get_CDR(params=cdr_params, comm=fenics_comm, local_uid=local_uid)
    results_list = cdr_fen(u=u)
    if verbose:
        print(f'Work done! Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
    simul_t1 = timetime()
    simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"
    if return_sim_time:
        return results_list, simul_time_str
    else:
        return results_list

def _run_experiment(cdr_params, u_input_idx_A, mesh_path, fenics_comm, mpi_rank, local_path_idx_A, local_uid, idx_A_str):
    make_directory(directory=local_path_idx_A,
                with_uid=False,
                with_datetime=False,
                return_new_directory=False,
                return_uid=False)

    results_list, simul_time_str = _run_cdr(fenics_comm=fenics_comm, 
                                            cdr_params=cdr_params, 
                                            u=u_input_idx_A,
                                            idx_A_str=idx_A_str,
                                            mpi_rank=mpi_rank, 
                                            local_uid=local_uid, 
                                            verbose=True, 
                                            return_sim_time=True)
    #store results to file
    #-----------------
    #store input data
    np_save(file=f"{local_path_idx_A}/input_data.npy", arr=u_input_idx_A)
    #store output data
    _store_output_data(fenics_comm, results_list, local_path_idx_A, mesh_path)
    #store local_uid, core rank, index of input_A, u_input data (redundancy), simulation time of this 1 iteration
    content_to_write_to_txt_dict = {'local_uid': local_uid,
                                    'rank': mpi_rank,
                                    'idx_A': idx_A_str,
                                    'input_data': u_input_idx_A,
                                    'simulation_time': simul_time_str}
    write_to_textfile(directory=local_path_idx_A, 
                    file_name='meta_data',
                    content_to_write_to_txt_dict=content_to_write_to_txt_dict,
                    include_current_datetime=True)
    
def cdr_vecSob_experiment(index_set_to_calculate, 
                        cdr_params, mpi_rank, 
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
    then define a generator by calling f=get_CDR(), 
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

    u_A = generate_data('log_uniform', min_u=5.5e11, max_u=1.5e12, size=3)
    u_E = generate_data('log_uniform', min_u=1.5e3, max_u=9.5e3, size=3)
    u_T_i = generate_data('uniform', min_u=850, max_u=1000, size=3)
    u_T_o = generate_data('uniform', min_u=200, max_u=400, size=3)
    u_phi = generate_data('uniform', min_u=0.5, max_u=1.5, size=3)

    u_all_realizations = np_arr([u_A, u_E, u_T_i, u_T_o, u_phi])
    u_I = u_all_realizations[:,0]
    u_II = u_all_realizations[:,1]
    u_III = u_all_realizations[:,2] 
    mesh_path = cdr_params["mesh_directory"]
    _run_experiment(cdr_params = cdr_params, 
                    u_input_idx_A=u_I, 
                    mesh_path=mesh_path, 
                    fenics_comm=fenics_comm, 
                    mpi_rank=mpi_rank, 
                    local_path_idx_A=f"{local_directory}/u_I", 
                    local_uid=local_uid, 
                    idx_A_str='I')
    _run_experiment(cdr_params = cdr_params, 
                    u_input_idx_A=u_II, 
                    mesh_path=mesh_path, 
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
        _run_experiment(cdr_params = cdr_params, 
                    u_input_idx_A=u_tilde, 
                    mesh_path=mesh_path, 
                    fenics_comm=fenics_comm, 
                    mpi_rank=mpi_rank, 
                    local_path_idx_A=f"{local_directory}/{idx_A_str}", 
                    local_uid=local_uid, 
                    idx_A_str=idx_A_str)