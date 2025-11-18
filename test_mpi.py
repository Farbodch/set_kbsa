from mpi4py import MPI
from datetime import datetime
import time, uuid
from os import makedirs
from kbsa.utils_kbsa import _get_u_indexSuperset_oneHot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dim_of_U=5
#all workers need to have access to this
u_indexSuperset_oneHot = _get_u_indexSuperset_oneHot(dim_of_U=dim_of_U, higher_order=False)
index_superset_len = len(u_indexSuperset_oneHot) 

N = 10 #number of total experiments to run

def work(str_to_write, rank, parent_path):
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
    IMPORTANT: the functions data WITHOUT the .txt file info is USELESS for SA!
    """
    local_uid = str(uuid.uuid4().hex)

    with open(f"{parent_path}/{local_uid}.txt", 'w') as f:
        f.write(f'rank_{rank}\n{str_to_write}')
    return 'done'

if rank == 0:
    parent_uid = str(uuid.uuid4().hex)
    parent_folder = f"test_mpi/{parent_uid}"
    makedirs(parent_folder, exist_ok=True)
else:
    parent_uid = None 

parent_uid = comm.bcast(parent_uid, root=0)
parent_folder = f"test_mpi/{parent_uid}"
indices = range(rank, N, size)
print(indices)
local_results = [(i, work(u_indexSuperset_oneHot[i], rank=rank, parent_path=parent_folder), parent_folder) for i in indices]
all_results = comm.gather(local_results, root=0)

if rank == 0:
    flat = [x for sub in all_results for x in sub]
    print(all_results)
    print("Done:", len(flat), "tasks")