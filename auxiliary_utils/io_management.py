from datetime import datetime
from uuid import uuid4
from os import makedirs
from pathlib import Path
import ast
from re import search as re_search
from typing import Union, Tuple, List
from dolfin import Mesh, FunctionSpace, Function, XDMFFile, MPI as dolfin_MPI
from numpy import (arange as np_arange,
                   zeros as np_zeros,
                   load as np_load,
                   array as np_array,
                   sum as np_sum,)
from auxiliary_utils.mpi_management import adjust_sampling_number_for_hsic
from numpy.random import (shuffle as np_shuffle)
from numpy.typing import NDArray
import pandas as pd

def make_directory(directory: str, 
                with_uid: bool = False,
                with_datetime: bool = False, 
                return_new_directory: bool = False, 
                return_uid = False) -> Union[str, Tuple[str, str]]:
    """Create a directory with (or without) a unique-id (uid) and current date and time (concatenated to the uid, if selected).

    Args:
        directory (str): parent directory to make the new directory(ies) in.
        with_uid (bool, optional): create new directory with a unique-id. Defaults to False.
        with_datetime (bool, optional): create new directory with current date and time (can be used with with_uid). Defaults to False.
        return_new_directory (bool, optional): if True, return the newly created direct. Defaults to False.
        return_uid (bool, optional): if True, return the unique-id, or datetime+unique-id, if selected. Defaults to False.

    Returns
    -------
        new_directory OR uid : str 
            if only return_new_directory OR return_uid set to True.

        new_directory AND uid : tuple[str, str] 
            if BOTH return_new_directory AND return_uid set to True.
    """
    uid = ''
    if with_datetime or with_uid:
        if with_uid:
            uid = str(uuid4().hex)
        if with_datetime:
            uid = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + with_uid*'__' + uid
        directory_to_generate = f"{directory}/{uid}"
    else:
        directory_to_generate = directory
    makedirs(directory_to_generate, exist_ok=True)
    if return_new_directory and return_uid:
        return directory_to_generate, uid
    elif return_new_directory and not return_uid:
        return directory_to_generate
    elif not return_new_directory and return_uid:
        return uid
    
def write_to_textfile(directory: str, 
                    file_name: str='meta_data', 
                    content_to_write_to_txt_dict: dict={'lorem': 'ipsum'},
                    include_current_datetime: bool=False):
    with open(f"{directory}/{file_name}.txt", 'w') as f:
        for key, value in content_to_write_to_txt_dict.items():
            f.write(f'{key}_{value};\n')
        if include_current_datetime:
            curr_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            f.write(f'datetime_{curr_datetime};\n')

#-------------------------------------
# functions to read-in results data (refactor?)
#-------------------------------------
def parse_metadata_file(metadata_path: Path) -> dict:
    parsed_metadata_dict = {}
    with open(metadata_path, 'r') as f:
        content = f.read()
    items = [item.strip() for item in content.split(';') if item.strip()]
    for item in items:
        if ':' in item:
            key, val = item.split(':', 1)
            parsed_metadata_dict[key.strip()] = val.strip()
        else:
            if item.startswith('num_of_grid_points'):
                parsed_metadata_dict['num_of_grid_points'] = item.replace('num_of_grid_points', '').strip()
            elif item.startswith('parent_uid'):
                parsed_metadata_dict['parent_uid'] = item.strip()
    return parsed_metadata_dict
def metadata_checker(metadata_path: Path, expected_metadata: dict) -> bool:
    if not metadata_path.exists():
        return False
    parsed_metadata_dict = parse_metadata_file(metadata_path)
    # first converting expected_metadata dict values to string for safe comparison,
    # then comparing.
    # we check only the keys passed-in through the expected_metadata dict.
    for key, expected_value in expected_metadata.items():
        if parsed_metadata_dict.get(key) != str(expected_value):
            return False
    return True
def read_in_results_data(base_directory: str, expected_metadata: dict):
    base_path = Path(base_directory)
    aggregated_data = []
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        metadata_path = subdir / 'meta_data.txt'
        results_path = subdir / 'results.txt'
        if metadata_path.exists() and results_path.exists():
            if metadata_checker(metadata_path=metadata_path, expected_metadata=expected_metadata):
                parsed_meta = parse_metadata_file(metadata_path)
                n_value = float(parsed_meta.get('num_of_outer_loop_sampling_n', 0))
                with open(results_path, 'r') as f:
                    results_str = f.read().strip()
                    try:
                        results_dict = ast.literal_eval(results_str)
                    except (SyntaxError, ValueError):
                        print(f"Failed to parse results in {subdir}")
                        continue
                record = {'num_of_outer_loop_sampling_n': n_value}
                record.update(results_dict)
                
                aggregated_data.append(record)

    df = pd.DataFrame(aggregated_data)
    if not df.empty:
        df = df.sort_values(by='num_of_outer_loop_sampling_n').reset_index(drop=True)
        
    return df
#-------------------------------------


#NOTe: This io manager for file directories has too many hard-coded condition checks. Not usable for, e.g., comsol structures or some other filing 
# structs with differet numbers of dirs, subdirs and subsubdirs. Change to generalize or remove redunandancy checks unless cdr or something? 
def get_data_file_directories(base_dir: str, 
                              data_type: str, 
                              process_type: str='cdr', 
                              return_n_max_list: bool=False,
                              num_of_mesh_steps: int = 128,
                              explicit_FEM_toggle: bool | None = None,
                              verbose=True,
                              enforce_params=True):
    """
    data_type ∈ {'fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data', 'diffusion_field'}
    Returns a list of file paths for chosen_cdr_field across all qualified sub_sub_directories.
    """

    '''TO DO: IMPORTANT REFACTOR -> MAKE ACCEPTABLE data_type AGNOSTIC AND NOT HARDCODED -> MAYBE RELY ON THE SETTINGS FILE?'''
    base_dir = Path(base_dir)   
    print(base_dir)
    if data_type not in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data', 'diffusion_field']:
        raise ValueError("chosen cdr field must be one of 'fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'input_data', 'diffusion_field'")

    target_filename = data_type
    if data_type in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field', 'diffusion_field']:
        target_filename += '.h5'
    else:
        target_filename += '.npy'
    # target_filename = f"{chosen_cdr_field}.h5"

    collected_paths = []
    num_of_parent_skips = 0
    num_of_sub_folder_skips = 0
    num_of_sub_sub_folder_skips = 0
    #loop over each parent directory inside base_dir
    for parent in sorted(base_dir.iterdir()):
        if not parent.is_dir():
            num_of_parent_skips += 1
            if verbose:
                print(f"Skipping {parent}: not a dir.")
            continue
        
        #1) check parent/meta_data.txt
        meta_file = parent / "meta_data.txt"
        if not meta_file.exists():
            num_of_parent_skips += 1
            if verbose:
                print(f"Skipping {parent}: no meta_data.txt")
            continue
        
        params = parse_meta_data(meta_file, process_type=process_type)
        if params is None:
            num_of_parent_skips += 1
            if verbose:
                print(f"Skipping {parent}: could not parse params.")
            continue
        num_of_mesh_steps_parent = params.get("num_of_mesh_steps")
        if num_of_mesh_steps_parent != num_of_mesh_steps:
            if verbose:
                print(f"Skipping {parent}: looking for mesh_steps of {num_of_mesh_steps}, found mesh_steps of {num_of_mesh_steps_parent}.")
            continue
        n_max_list = []
        ''' TO DO : These enforce_params values are arbitrary and need to be set in a higher-level settings file
                    by user, but now just check to enforce some abitrary values for this specific paper and 
                    simply fails to read in anything else.'''
        if 'diffusion_1d' in process_type:
            P = params.get("P")
        if enforce_params:
            if 'cdr' in process_type:
            #validate required conditions
                if not (
                    params.get("t_end") == 0.05 and
                    params.get("num_steps") == 500 and
                    params.get("return_bool") is False
                ):
                    num_of_parent_skips += 1
                    if verbose:
                        print(f"Skipping {parent}: incorrect params")
                    continue
            
            elif 'diffusion_1d' in process_type:
                if not (
                    P == 3 and
                    params.get("num_of_mesh_steps") == num_of_mesh_steps
                ):
                    num_of_parent_skips += 1
                    if verbose:
                        print(f"Skipping {parent}: incorrect params")
                    continue
            elif process_type == 'analytical' and return_n_max_list:
                n_max_list.append(params["total_num_of_experiments"])
        
        #2) check subdirectories
        sub_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])

        #skip if parent folder has no subfolder structure
        if 'cdr' in process_type:
            if len(sub_dirs) == 0:
                num_of_parent_skips += 1
                if verbose:
                    print(f"Skipping {parent}: no sub-directories found.")
                continue

        for sub in sub_dirs:
            sub_subs = sorted([d for d in sub.iterdir() if d.is_dir()])

            if 'cdr' in process_type:
                #NOTe: This sub_sub num of sub folders check is not static for vecSob, as it can have
                # a varying number depending on whether the required orders were created (for total vs
                # just the closed Sob indices). As such, a static folder cardinality check does not work here.
                if len(sub_subs) != 6 and process_type == 'cdr':
                    num_of_sub_folder_skips += 1
                    if verbose:
                        print(f"Skipping {parent}/{sub}: does not have 6 sub_sub-folders.")
                    continue

                #3)check inside each sub_sub_directory for exactly 10 files
                for sub_sub in sub_subs:
                    files = sorted([f for f in sub_sub.iterdir() if f.is_file()])
                    if len(files) != 10 and Path(sub_sub).name != 'u_III':
                        num_of_sub_sub_folder_skips += 1
                        if verbose:
                            print(f"Skipping {parent}/{sub}/{sub_sub}: does not contain the expected 10 files.")
                        continue
                    elif len(files) != 2 and Path(sub_sub).name == 'u_III':
                        num_of_sub_sub_folder_skips += 1
                        if verbose:
                            print(f"Skipping {parent}/{sub}/{sub_sub}: does not contain the expected 2 files.")
                        continue

                    #4.a)collect chosen .h5 or .npy file
                    wanted_path = sub_sub / target_filename
                    if wanted_path.exists():
                        if data_type == 'input_data':
                            collected_paths.append(str(wanted_path))
                        else:
                            collected_paths.append(str(wanted_path)[:-3])
                    else:
                        num_of_sub_sub_folder_skips += 1
                        if verbose:
                            print(f"Warning: {wanted_path} missing.")

            elif process_type == 'analytical':
                #4.b)collect chosen .h5 or .npy file
                wanted_path = sub / target_filename
                if wanted_path.exists():
                    if data_type == 'input_data':
                        collected_paths.append(str(wanted_path))
                    else:
                        collected_paths.append(str(wanted_path)[:-3])
                else:
                    num_of_sub_sub_folder_skips += 1
                    if verbose:
                        print(f"Warning: {wanted_path} missing.")
            elif 'diffusion_1d' in process_type:
                #NOTe: This sub_sub num of sub folders check is not static for vecSob, as it can have
                # a varying number depending on whether the required orders were created (for total vs
                # just the closed Sob indices). As such, a static folder cardinality check does not work here.
                if 'vecSob' in process_type:
                    if len(sub_subs) != 2*P+3:
                        num_of_sub_folder_skips += 1
                        if verbose:
                            print(f"Skipping {parent}/{sub}: does not have {P+1} sub_sub-folders.")
                        continue
                else:
                    if len(sub_subs) != P+1:
                        num_of_sub_folder_skips += 1
                        if verbose:
                            print(f"Skipping {parent}/{sub}: does not have {P+1} sub_sub-folders.")
                        continue

                #3)check inside each sub_sub_directory for exactly 4 files
                for sub_sub in sub_subs:
                    files = sorted([f for f in sub_sub.iterdir() if f.is_file()])
                    if len(files) != 4:
                        num_of_sub_sub_folder_skips += 1
                        if verbose:
                            print(f"Skipping {sub_sub}: does not contain the expected 4 files.")
                        continue
                    # elif len(files) != 2 and Path(sub_sub).name == 'u_III':
                    #     num_of_sub_sub_folder_skips += 1
                    #     if verbose:
                    #         print(f"Skipping {sub_sub}: does not contain the expected 2 files.")
                    #     continue

                    #4.a)collect chosen .h5 or .npy file
                    wanted_path = sub_sub / target_filename
                    if wanted_path.exists():
                        if data_type == 'input_data':
                            collected_paths.append(str(wanted_path))
                        else:
                            collected_paths.append(str(wanted_path)[:-3])
                    else:
                        num_of_sub_sub_folder_skips += 1
                        if verbose:
                            print(f"Warning: {wanted_path} missing.")
        print(f"Num_of_parent_skips: {num_of_parent_skips}")
        print(f"num_of_sub_folder_skips: {num_of_sub_folder_skips}")
        print(f"num_of_sub_sub_folder_skips: {num_of_sub_sub_folder_skips}")
        # if not parent_qualified:
        #     continue
    if return_n_max_list:
        return collected_paths, n_max_list
    
    return collected_paths

def parse_meta_data(meta_file: Path, process_type: str):
    """
    Extract and return params dict from meta_data.txt.
    """

    '''TO DO: REFACTOR TO USE model_params FOR ALL MODELS, NOT params_ and etc.'''
    text = meta_file.read_text()
    if 'cdr' in process_type:
    #params appear after "params_"
        start_idx = text.find("params_")
        if start_idx == -1:
            return None
        start_idx += len("params_")
        #extract param dictionary content between '{' and '}'
        dict_str = text[start_idx:].strip()
        dict_str = dict_str[dict_str.find("{"): dict_str.rfind("}")+1]
        try:
            params = ast.literal_eval(dict_str)
            return params
        except:
            print(f"Could not parse params in {meta_file}")
            return None
    elif 'diffusion_1d' in process_type:
    #model_params appear after "model_params_"
        start_idx = text.find("params_")
        if start_idx == -1:
            return None
        start_idx += len("params_")
        #extract param dictionary content between '{' and '}'
        dict_str = text[start_idx:].strip()
        dict_str = dict_str[dict_str.find("{"): dict_str.rfind("}")+1]
        try:
            params = ast.literal_eval(dict_str)
            return params
        except:
            print(f"Could not parse model_params in {meta_file}")
            return None
    elif process_type == 'analytical':
        meta_data = {}
        match = re_search(r"total_num_of_experiments_(\d+)", text)
        if match:
            total_num_of_experiments = int(match.group(1))
        else:
            total_num_of_experiments = None
        meta_data['total_num_of_experiments'] = total_num_of_experiments
        return meta_data
    else:
        raise ValueError("This should not be reached.")

def load_mesh(mesh_dir: str = "data/mesh_data/cdr/rectangle.xdmf"):
    fenics_comm = dolfin_MPI.comm_self
    mesh = Mesh(fenics_comm)
    with XDMFFile(fenics_comm, mesh_dir) as xdmf:
        xdmf.read(mesh)
    return mesh

def load_function_space(mesh: Mesh, cg_order=1):
    V = FunctionSpace(mesh, 'CG', cg_order) 
    return V

def load_fenics_function(field_file_path: str, mesh_dir: str="data/mesh_data/cdr/rectangle.xdmf"):
    """ this function takes in a path to the file where fenics function of the field_of_interest is located 
    and returns the associated "<class 'dolfin.function.function.Function'>", on a CG-1 elements. """
    field_of_interest = Path(field_file_path).name
    if field_of_interest not in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']:
        raise ValueError("cdr field must be one of 'fuel_field', 'oxygen_field', 'product_field', 'temp_field'")
    mesh = load_mesh(mesh_dir=mesh_dir)
    V_1 = FunctionSpace(mesh, 'CG', 1) 
    field_t_now = Function(V_1)
    with XDMFFile(mesh.mpi_comm(), field_file_path) as xdmf_1:
        xdmf_1.read_checkpoint(field_t_now, field_of_interest, 0)
    return field_t_now

def get_input_data_from_file_fenics_function(data_directory: Union[str, None] = None,
                                            num_of_u_inputs: int = 1,
                                            n: int = 1,
                                            shuffle: bool = False,
                                            return_directories: bool = False,
                                            adjust_for_mpi: bool = False,
                                            mpi_size: int = 1,
                                            process_type: str = 'cdr',
                                            input_data_directories_list_to_use: Union[List, None]=None) -> Union[NDArray, Tuple[NDArray, List]]:
    if input_data_directories_list_to_use is None:
        assert data_directory is not None, "Must pass in parent data_directory if no explicit input_data_directories_list_to_use is passed in."
        input_data_dirs_list = get_data_file_directories(data_directory, process_type=process_type, data_type='input_data')
        n_max = len(input_data_dirs_list)
        data_dir_indices = np_arange(0, n_max)
        if shuffle:
            np_shuffle(data_dir_indices)
        if n > n_max:
            n = n_max
        if adjust_for_mpi:
            n = adjust_sampling_number_for_hsic(n=n,
                                                n_max=n_max,
                                                size=mpi_size,
                                                verbose=True)
        if n < n_max:
            data_dir_indices = data_dir_indices[:n]
        
        #if shuffle was toggled, then we shuffle based on indices and then add the data file directories here
        input_data_directories_list_to_use = [input_data_dirs_list[i] for i in data_dir_indices[:n]]
    u_arr = np_zeros((n, num_of_u_inputs))
    for i, dir in enumerate(input_data_directories_list_to_use):
        u_arr[i] = np_load(dir)
    if return_directories:
        return u_arr, input_data_directories_list_to_use
    return u_arr

def get_input_data_from_file_analytical(data_directory: str,
                                        num_of_u_inputs: int, 
                                        n: int,
                                        shuffle: bool = False,
                                        return_directories: bool = False) -> NDArray | tuple[NDArray, List]:
    data_directories, n_max_list = get_data_file_directories(data_directory, 
                                            data_type='input_data', 
                                            process_type='analytical',
                                            return_n_max_list=True)
    n_max = np_sum(np_array(n_max_list))
    if type(n) is not int:
        n = int(n)
    if n > n_max:
        n = n_max
    if n < 1:   
        n=1
    u_arr = np_zeros((n, num_of_u_inputs))
    u_total_counter = 0
    if shuffle:
        np_shuffle(data_directories)
    for i, dir in enumerate(data_directories):
        if u_total_counter == n:
            continue
        u_arr_curr = np_load(dir)
        left_to_add_counts = n - u_total_counter

        if left_to_add_counts >= len(u_arr_curr):
            u_arr[u_total_counter:(u_total_counter+len(u_arr_curr))] = u_arr_curr
            u_total_counter += len(u_arr_curr)
        else:
            u_arr[u_total_counter:(u_total_counter+left_to_add_counts)] = u_arr_curr[:left_to_add_counts]
            u_total_counter += left_to_add_counts
    if return_directories:
        return u_arr, data_directories[:i]
    return u_arr

def write_from_dict_to_text_file(data_to_write_dict: dict,
                                 data_file_name: str = '',
                                 data_parent_directory: str | None = None,
                                 generate_new_uid: bool = False,
                                 data_directory_with_uid: str | None = None, 
                                 data_uid: str | None = None,
                                 return_data_directory_with_uid: bool = False):
    assert data_file_name != '', "data_file_name required but NOT passed in."
    if generate_new_uid:
        assert data_parent_directory is not None, f"When generate_new_uid is set to True, must pass in a sup_directory but now {data_parent_directory} was passed in."
        data_uid = datetime.now().strftime("%Y_%m_%d_%H_%M_%S__") + str(uuid4().hex)
        data_directory_with_uid = f"{data_parent_directory}/{data_uid}"

    assert data_directory_with_uid is not None, "Both the directory data_directory_with_uid was passed in as None, and generate_new_uid was set to False! Need either of the two."
    assert data_uid in data_directory_with_uid, f"Local data directory {data_directory_with_uid} should be, but is NOT being identified with UID {data_uid}."

    makedirs(data_directory_with_uid, exist_ok=True)
    with open(f"{data_directory_with_uid}/{data_file_name}.txt", 'w') as f:
        f.write(f'data_uid_{data_uid};\n')
        for key, val in data_to_write_dict.items():
            f.write(f'{key}:{val};\n')
    return data_directory_with_uid