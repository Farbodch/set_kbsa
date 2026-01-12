from auxiliary_utils.io_management import load_mesh
from numpy.linalg import norm as np_norm
from scipy.spatial import cKDTree
from dolfin.cpp.mesh import edges

#minsup_{cells \in mesh}(cell_diameter), cell diameter wrt all cells in mesh
def get_min_cell_diam_of_mesh(mesh=None, mesh_directory: str=None) -> float:
    if mesh is None and mesh_directory is None:
        raise ValueError("Either a mesh object or a directory to mesh must be passed in. None were provided.")
    if mesh is None:
        mesh = load_mesh(mesh_dir=mesh_directory)
    return mesh.hmin()

#minimum vertex-to-vertex distance (not necessarily an edge in the mesh)
def get_min_vert_to_vert_dist_of_mesh(mesh=None, mesh_directory: str=None) -> float:
    if mesh is None and mesh_directory is None:
        raise ValueError("Either a mesh object or a directory to mesh must be passed in. None were provided.")
    if mesh is None:
        mesh = load_mesh(mesh_dir=mesh_directory)
    coords = mesh.coordinates()
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(dists[:, 1].min())

#length of the smallest edge in the mesh
def get_min_edge_of_mesh(mesh=None, mesh_directory: str=None) -> float:
    if mesh is None and mesh_directory is None:
        raise ValueError("Either a mesh object or a directory to mesh must be passed in. None were provided.")
    if mesh is None:
        mesh = load_mesh(mesh_dir=mesh_directory)
    coords = mesh.coordinates()
    return min(np_norm(coords[e.entities(0)[0]] - coords[e.entities(0)[1]]) for e in edges(mesh)) 

def get_shortest_geom_mesh_dist(mesh=None, mesh_directory: str=None, verbose: bool=False) -> float:
    if mesh is None and mesh_directory is None:
        raise ValueError("Either a mesh object or a directory to mesh must be passed in. None were provided.")
    if mesh is None:
        mesh = load_mesh(mesh_dir=mesh_directory)
    min_cell_diam_of_mesh = get_min_cell_diam_of_mesh(mesh=mesh)
    min_vert_to_vert_dist_of_mesh = get_min_vert_to_vert_dist_of_mesh(mesh=mesh)
    min_edge_of_mesh = get_min_edge_of_mesh(mesh=mesh)
    
    idx, smallest_geom_dist = min(enumerate([min_cell_diam_of_mesh, min_vert_to_vert_dist_of_mesh, min_edge_of_mesh]), key=lambda x: x[1])

    if verbose:
        name_table = ['min_cell_diam_of_mesh', 'min_vert_to_vert_dist_of_mesh', 'min_edge_of_mesh']
        print(f"Found {name_table[idx]} = {smallest_geom_dist} as the shortest distance in mesh.")
        
    return smallest_geom_dist