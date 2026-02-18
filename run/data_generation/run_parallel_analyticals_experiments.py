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
from data_generation_scripts.analytical_models import analytical_models_experiment
from auxiliary_utils.io_management import make_directory, write_to_textfile
from numpy import (pi as np_pi, floor as np_floor)
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10, required=True, help='Number of input data to generate for a given analytical function.')
    parser.add_argument("--model", type=str, default='ishigami', required=True, help='The analytical model for which the data is being generated.')
    parser.add_argument("--bins", type=int, default=1, help='Number of bins to chunk the N data into. E.g., N=1000 and chunk=4, will create 4 folders, each with a .npy file containing 250 data points. In case (N mod chunk) != 0, the first bin will contain the remainder.')
    user_inputs = parser.parse_args()
    N = user_inputs.N #number of total experiments to run
    model_name = user_inputs.model
    bins = user_inputs.bins

    if model_name == 'ishigami':
        u_domain_specs = [{'distribution_type': 'uniform', 'min': -np_pi, 'max': np_pi},
                        {'distribution_type': 'uniform', 'min': -np_pi, 'max': np_pi}]
    elif model_name == 'fellmann':
        u_domain_specs = [{'distribution_type': 'uniform', 'min': -5, 'max': 5},
                        {'distribution_type': 'uniform', 'min': -5, 'max': 5}]

    parent_directory, parent_uid = make_directory(directory=f'data/experiment_data/{model_name}',
                                                    with_uid=True,
                                                    with_datetime=True, 
                                                    return_new_directory=True, 
                                                    return_uid=True)
    total_N = N
    if bins <= 1:
        bins = 1
    if bins > 1:
        remainder_in_first_bin = int(N % bins)
        N = int(np_floor(N / bins))
        analytical_models_experiment(u_domain_specs=u_domain_specs,
                                    num_of_experiments=N+remainder_in_first_bin,
                                    parent_directory=parent_directory,
                                    model_name=model_name)
    for _ in range(bins-1):
        analytical_models_experiment(u_domain_specs=u_domain_specs,
                                    num_of_experiments=N,
                                    parent_directory=parent_directory,
                                    model_name=model_name)
    
    content_to_write_to_txt_dict = {'parent_uid': parent_uid,
                                    'total_num_of_experiments': int(total_N),
                                    'model_name': model_name,
                                    'u_domain_specs': u_domain_specs}
    write_to_textfile(directory=parent_directory,
                        file_name='meta_data',
                        content_to_write_to_txt_dict=content_to_write_to_txt_dict)
if __name__ == "__main__":
    main()