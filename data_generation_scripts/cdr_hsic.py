from time import time as timetime
from kbsa.model_generators import get_CDR
from numeric_models.numeric_models_utils import generate_data
from numpy import (array as np_arr, save as np_save)
from dolfin import XDMFFile, Mesh, MPI as dolfin_MPI
from auxiliary_utils.io_management import make_directory, write_to_textfile

def cdr_hsic_experiment(u_indexSuperset_oneHot, 
                        cdr_params, mpi_rank, 
                        parent_directory, 
                        make_directory_with_uid: bool=True,
                        make_directory_with_datetime: bool=False):
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
    IMPORTANT: the functions data without the .txt file info is useless for SA!
    """
    fenics_comm = dolfin_MPI.comm_self

    local_directory, local_uid = make_directory(directory=parent_directory,
                                                with_uid=make_directory_with_uid,
                                                with_datetime=make_directory_with_datetime, 
                                                return_new_directory=True, 
                                                return_uid=True)

    u_A = generate_data('log_uniform', min_u=5.5e11, max_u=1.5e12, size=6)
    u_E = generate_data('log_uniform', min_u=1.5e3, max_u=9.5e3, size=6)
    u_T_i = generate_data('uniform', min_u=850, max_u=1000, size=6)
    u_T_o = generate_data('uniform', min_u=200, max_u=400, size=6)
    u_phi = generate_data('uniform', min_u=0.5, max_u=1.5, size=6)

    u_input_all_A = np_arr([u_A, u_E, u_T_i, u_T_o, u_phi])
    mesh_path = cdr_params["mesh_2D_dir"]
    for idx_A, idx_A_str in enumerate(u_indexSuperset_oneHot):
        local_path_idx_A = f"{local_directory}/{idx_A_str}"
        make_directory(directory=local_path_idx_A,
                    with_uid=False,
                    with_datetime=False,
                    return_new_directory=False,
                    return_uid=False)

        u_input_idx_A = u_input_all_A[:, idx_A]
        simul_t0 = timetime()
        print(f'Started work. Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
        cdr_fen = get_CDR(params=cdr_params, comm=fenics_comm, local_uid=local_uid)
        results_list = cdr_fen(u=u_input_idx_A)
        print(f'Work done! Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
        simul_t1 = timetime()
        simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"
        
        #store results to file
        #-----------------
        #store input data
        np_save(file=f"{local_path_idx_A}/input_data.npy", arr=u_input_idx_A)
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