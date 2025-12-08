import numpy as np
from os import makedirs
from uuid import uuid4

def save_input_data(save_directory, input_data):
    makedirs(save_directory, exist_ok=True)
    np.save(file=f"{save_directory}/input_data.np", arr=input_data)

def generate_local_dir_with_uid(parent_directory):
    local_uid = str(uuid4().hex)
    local_directory = f"{parent_directory}/{local_uid}"
    makedirs(local_directory, exist_ok=True)

def save_to_txt_file(data_to_save, file_name, save_directory):
    with open(f"{save_directory}/{file_name}.txt", 'w') as f:
        f.write(data_to_save)