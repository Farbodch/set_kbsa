from numeric_models.numeric_models_utils import generate_data
from numpy import (save as np_save, zeros as np_zeros)
from auxiliary_utils.file_management import make_directory, write_to_textfile
from time import time as timetime

def analytical_models_experiment(u_domain_specs: list, 
                                num_of_experiments: int, 
                                parent_directory: str, 
                                model_name: str=None, 
                                mpi_rank: int=None):
    """
    Here, we just need to generate random inputs and save them to file, and save a meta_data file.
    Need to adjust both this and the cdr.py data_generation script to allow for the PickFreeze-style 
    data generation as well.
    """
    simul_t0 = timetime()
    u = np_zeros(shape=(num_of_experiments, len(u_domain_specs)))
    for i, u_i_specs_dict in enumerate(u_domain_specs): 
        distribution_type = u_i_specs_dict['distribution_type']
        min_u = u_i_specs_dict['min']
        max_u = u_i_specs_dict['max']
        u[:,i] = generate_data(distribution_type=distribution_type, min_u=min_u, max_u=max_u, size=num_of_experiments)
    simul_t1 = timetime()
    simul_time_str = f"simulation_time(s):{simul_t1 - simul_t0:.6f}"

    local_directory, local_uid = make_directory(directory=parent_directory,
                                            with_uid=True,
                                            with_datetime=False, 
                                            return_new_directory=True, 
                                            return_uid=True)

    np_save(file=f"{local_directory}/input_data.npy", arr=u)

    content_to_write_to_txt_dict = {'model_name': model_name,
                                    'local_uid': local_uid,
                                    'rank': mpi_rank,
                                    'num_of_experiments':num_of_experiments,
                                    'simulation_time': simul_time_str,
                                    'u_domain_specs': u_domain_specs}

    #pop info that remain "None" from content_to_write_to_txt_dict
    content_to_write_to_txt_dict = {key: val for key, val in content_to_write_to_txt_dict.items() if val is not None}

    if num_of_experiments == 1:
        content_to_write_to_txt_dict['input_data'] = u

    write_to_textfile(directory=local_directory, 
                        file_name='meta_data',
                        content_to_write_to_txt_dict=content_to_write_to_txt_dict,
                        include_current_datetime=True)