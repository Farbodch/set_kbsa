from datetime import datetime
from uuid import uuid4
from os import makedirs

def make_directory(directory: str, 
                with_uid: bool = False,
                with_datetime: bool = False, 
                return_new_directory: bool = False, 
                return_uid = False) -> str | tuple[str, str]:
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