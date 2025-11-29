from pathlib import Path
import ast
from dolfin import Mesh, FunctionSpace, Function, XDMFFile
from numpy import ndarray, float32, uint8, fill_diagonal, array as np_array
from numpy import zeros as np_zeros
from numpy import arange as np_arange
from numpy.random import shuffle as np_shuffle
from numpy.random import uniform as np_unif
import gc

def parse_meta_data(meta_file: Path):
    """
    Extract and return cdr_params dict from meta_data.txt.
    """
    text = meta_file.read_text()
    #cdr_params appear after "cdr_params_"
    start_idx = text.find("cdr_params_")
    if start_idx == -1:
        return None
    start_idx += len("cdr_params_")
    #extract param dictionary content between '{' and '}'
    dict_str = text[start_idx:].strip()
    dict_str = dict_str[dict_str.find("{"): dict_str.rfind("}")+1]

    try:
        params = ast.literal_eval(dict_str)
        return params
    except:
        print(f"Could not parse cdr_params in {meta_file}")
        return None

def get_field_file_dirs(base_dir: str, chosen_cdr_field: str):
    """
    chosen_cdr_field ∈ {'fuel_field', 'oxygen_field', 'product_field', 'temp_field'}
    Returns a list of file paths for chosen_cdr_field across all qualified sub_sub_directories.
    """
    base_dir = Path(base_dir)   

    if chosen_cdr_field not in ['fuel_field', 'oxygen_field', 'product_field', 'temp_field']:
        raise ValueError("chosen cdr field must be one of 'fuel_field', 'oxygen_field', 'product_field', 'temp_field'")

    target_filename = f"{chosen_cdr_field}.h5"
    collected_paths = []

    num_of_parent_skips = 0
    num_of_sub_folder_skips = 0
    num_of_sub_sub_folder_skips = 0
    #loop over each parent directory inside base_dir
    for parent in base_dir.iterdir():
        if not parent.is_dir():
            num_of_parent_skips += 1
            continue

        #1) check parent/meta_data.txt
        meta_file = parent / "meta_data.txt"
        if not meta_file.exists():
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: no meta_data.txt")
            continue

        params = parse_meta_data(meta_file)
        if params is None:
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: could not parse params.")
            continue

        #validate required conditions
        if not (
            params.get("t_end") == 0.05 and
            params.get("num_steps") == 500 and
            params.get("return_bool") is False
        ):
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: incorrect cdr_params")
            continue

        #2) check subdirectories
        sub_dirs = [d for d in parent.iterdir() if d.is_dir()]

        #skip if parent folder has no subfolder structure
        if len(sub_dirs) == 0:
            num_of_parent_skips += 1
            # print(f"Skipping {parent}: no sub-directories found.")
            continue

        for sub in sub_dirs:
            sub_subs = [d for d in sub.iterdir() if d.is_dir()]
            if len(sub_subs) != 6:
                num_of_sub_folder_skips += 1
                # print(f"Skipping {parent}/{sub}: does not have 6 sub_sub-folders.")
                continue

            #3)check inside each sub_sub_directory for exactly 9 files
            for sub_sub in sub_subs:
                files = [f for f in sub_sub.iterdir() if f.is_file()]
                if len(files) != 10:
                    num_of_sub_sub_folder_skips += 1
                    # print(f"Skipping {parent}/{sub}/{sub_sub}: does not contain 9 files.")
                    continue

                #4)collect chosen H5 file
                wanted_path = sub_sub / target_filename
                if wanted_path.exists():
                    collected_paths.append(str(wanted_path)[:-3])
                else:
                    num_of_sub_sub_folder_skips += 1
                    print(f"Warning: {wanted_path} missing.")
                    
        print(f"Num_of_parent_skips: {num_of_parent_skips}")
        print(f"num_of_sub_folder_skips: {num_of_sub_folder_skips}")
        print(f"num_of_sub_sub_folder_skips: {num_of_sub_sub_folder_skips}")
        # if not parent_qualified:
        #     continue

    return collected_paths


def load_mesh(mesh_dir: str = "data/CDR/mesh_save_dir/rectangle.xdmf"):
    mesh = Mesh()
    with XDMFFile(mesh_dir) as xdmf:
        xdmf.read(mesh)
    return mesh
def load_function_space(mesh: Mesh, cg_order=1):
    V = FunctionSpace(mesh, 'CG', cg_order) 
    return V

def load_fenics_function(field_file_path: str, mesh_dir: str="data/CDR/mesh_save_dir/rectangle.xdmf"):
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

def get_mesh_bounds(mesh):
    mesh_coords = mesh.coordinates()
    xmin, ymin = mesh_coords.min(axis=0)
    xmax, ymax = mesh_coords.max(axis=0)
    return (xmin, xmax), (ymin, ymax)

def sample_fenics_function(field_file_path: str, 
                        mesh: Mesh,
                        func_space_V: FunctionSpace,
                        test_domain: ndarray = np_array([[0,1],[0,0.5]]), 
                        num_of_spatial_sampling_m: int = 5,
                        return_as_bool: bool = False,
                        g_constraint: float = 0):
    m = num_of_spatial_sampling_m
    test_domain = np_array(test_domain)
    
    x_vect = np_unif([test_domain[0,0], test_domain[1,0]], [test_domain[0,1], test_domain[1,1]], size=(m, 2))
    field_of_interest = Path(field_file_path).name
    
    f = Function(func_space_V)
    with XDMFFile(mesh.mpi_comm(), field_file_path) as xdmf_1:
        xdmf_1.read_checkpoint(f, field_of_interest, 0)

    f_samplings = np_zeros(m)

    for i, x in enumerate(x_vect):
        f_samplings[i] = f(x[0], x[1])

    del f
    gc.collect()
    
    if return_as_bool:
        return (f_samplings <= g_constraint).astype(uint8)
    else:
        return f_samplings

def approximate_k_set(lambda_matrix: ndarray,
                    lambda_X: float) -> ndarray:
    _, m = lambda_matrix.shape

    #row sums /w output shape (n,)
    just_ones_count = lambda_matrix.sum(axis=1)

    #pairwise AND count /w output shape (n,n)
    and_count = lambda_matrix @ lambda_matrix.T

    #xOr count using identity AxOrB=a+b-2(aANDb)
    xOr_count = just_ones_count[:, None] + just_ones_count[None, :] - 2*and_count
    xOr_count = xOr_count.astype(float32)

    gamma_matrix = lambda_X*(xOr_count/m)
    fill_diagonal(gamma_matrix, 0.0)

    return gamma_matrix


def get_gamma_matrix(data_dir_list_of_fields: list, 
                    n: int, 
                    num_of_spatial_sampling_m: int,
                    mesh: Mesh,
                    func_space_V: FunctionSpace,
                    test_domain: ndarray = np_array([[0,1],[0,0.5]]),
                    return_as_bool: bool = False,
                    g_constraint: float = 0):
    
    m = num_of_spatial_sampling_m
    n_max = len(data_dir_list_of_fields)
    data_dir_indices = np_arange(0, n_max)
    np_shuffle(data_dir_indices)
    if n < n_max:
        data_dir_indices = data_dir_indices[:n]
    else:
        n = n_max

    lambda_matrix = np_zeros((n, m), dtype=uint8)
    for i, data_dir_idx in enumerate(data_dir_indices):
        lambda_matrix[i] = sample_fenics_function(field_file_path=data_dir_list_of_fields[data_dir_idx],
                                                mesh=mesh,
                                                func_space_V=func_space_V,
                                                test_domain=test_domain,
                                                num_of_spatial_sampling_m=m,
                                                return_as_bool=return_as_bool,
                                                g_constraint=g_constraint)
    
    lambda_X = 1
    for i in range(len(test_domain)):
        lambda_curr_dom = max(test_domain[i])-min(test_domain[i])
        if lambda_curr_dom != 0:
            lambda_X *= lambda_curr_dom
    
    gamma_matrix = approximate_k_set(lambda_matrix=lambda_matrix, lambda_X=lambda_X)

    return gamma_matrix

"""
NOT!E: NEED TO TAKE THE INDICES OF DIRS OUT SINCE enteries of the outputs matrix k_gamma needs to 
be able to be associated with the inputs matrix enteries K_u as each i,j. Also, port the port the vectorized
_build_K_sob to this decoupled case. 
To Do:
-> vectorize or MPI the k_gamma implementation?
-> MPI the ported version of the _build_K_sob_vectorized
"""