from time import time as timetime
from numeric_models.pde_models import get_1D_diff_FEM
from numeric_models.numeric_models_utils import generate_data
from numpy import (array as np_arr, save as np_save)
from dolfin import Mesh, XDMFFile, MPI as dolfin_MPI
from auxiliary_utils.io_management import make_directory, write_to_textfile

def diffusion_1d_hsic_experiment(u_indexSuperset_oneHot, 
                        params, mpi_rank, 
                        parent_directory, 
                        make_directory_with_uid: bool=True,
                        make_directory_with_datetime: bool=False):
    """
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
    # choosing size=depth_P for individual indices for each randomized-input 
    # (P_1 ...01, P_2 ...10, P_3 ...100, etc.) 
    # + 1 for all (11...111) = depth_P + 1
    # u_P = generate_data('uniform', min_u=-1, max_u=1, size=depth_P+1)

    u_input_all_A = np_arr([generate_data('uniform', min_u=-1, max_u=1, size=depth_P+1) for _ in range(depth_P)])
    mesh_directory = params["mesh_directory"]
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

        diffusion_fen = get_1D_diff_FEM(params=params, comm=fenics_comm, local_uid=local_uid)
        results = diffusion_fen(u=u_input_idx_A)
        print(f'Work done! Rank {mpi_rank} - A_str - {idx_A_str} - local uid {local_uid}', flush=True)
        simul_t1 = timetime()
        simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"
        
        #-----------------
        # store results to file
        #-----------------
        # store input data
        np_save(file=f"{local_path_idx_A}/input_data.npy", arr=u_input_idx_A)
        # store output data
        mesh = Mesh(fenics_comm)
        with XDMFFile(fenics_comm, mesh_directory) as xdmf:
            xdmf.read(mesh)
        with XDMFFile(fenics_comm, f'{local_path_idx_A}/diffusion_field') as xdmf:
            xdmf.write_checkpoint(results, 'diffusion_field', 0, XDMFFile.Encoding.HDF5, append=False)

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