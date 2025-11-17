from mpi4py import MPI
from datetime import datetime
import time, uuid
from os import makedirs
from kbsa.utils_kbsa import _get_u_indexSuperset_oneHot
import warnings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dim_of_U=5
u_indexSuperset_oneHot = _get_u_indexSuperset_oneHot(dim_of_U=dim_of_U, higher_order=False)
N = len(u_indexSuperset_oneHot)

def work(str_to_write, rank, parent_path):
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