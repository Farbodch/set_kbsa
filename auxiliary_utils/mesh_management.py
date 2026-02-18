from auxiliary_utils.io_management import load_mesh
from numpy.linalg import norm as np_norm 
from numpy import (unique as np_unique,
                    argmin as np_argmin,
                    abs as np_abs,
                    all as np_all,
                    any as np_any,
                    ones as np_ones,
                    ceil as np_ceil)
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from dolfin.cpp.mesh import edges
from dolfin import (MeshFunction,
                    FunctionSpace,
                    cells,
                    Mesh as df_Mesh,
                    IntervalMesh as df_IntervalMesh,
                    inner as df_inner,
                    Measure as df_Measure,
                    assemble as df_assemble,
                    TrialFunction as df_TrialFunction,
                    TestFunction as df_TestFunction,
                    as_backend_type as df_as_backend_type,
                    XDMFFile as df_XDMFFile)
from dolfin.cpp.mesh import MeshFunctionSizet
import gmsh
import meshio
from os import (makedirs as os_makedirs, 
                path as os_path)
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

def _snap_1d_coords_to_mesh_nodes(coords_1d, lower_bound, upper_bound, mode):
    """
    coords_1d: 1D numpy array of mesh vertex coordinates in a dimension (global, may contain repeats)
    interval_min, interval_max: float test_domain, interval_min<=interval_max
    mode: 'lower'|'upper'|'nearest'
       - lower: low_s = smallest node coord >= lower_bound, high_s = largest node coord <= upper_bound
       - upper: low_s = largest  node coord <= lower_bound, high_s = smallest node coord >= upper_bound
       - nearest: low_s = nearest node coord to lower_bound, high_s = nearest node coord to upper_bound

    Returns (low_s, high_s) snapped test_domain.
    """
    xs = np_unique(coords_1d)
    xs.sort()

    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    if mode == "lower":
        # Inside-closest: shrink interval to fit within [interval_min, interval_max]
        candidate_min_verts = xs[xs >= lower_bound]
        candidate_max_verts = xs[xs <= upper_bound]
        if len(candidate_min_verts) == 0 or len(candidate_max_verts) == 0:
            # Interval doesn't intersect node coordinates meaningfully; fallback to nearest
            mode = "nearest"
        else:
            low_s = candidate_min_verts[0]
            high_s = candidate_max_verts[-1]
            if low_s > high_s:
                # No node strictly inside both constraints; fallback to nearest
                mode = "nearest"
            else:
                return float(low_s), float(high_s)

    if mode == "upper":
        # Outside-closest: expand interval to cover [lower_bound, upper_bound]
        candidate_min_verts = xs[xs <= lower_bound]
        candidate_max_verts = xs[xs >= upper_bound]
        if len(candidate_min_verts) == 0:
            low_s = xs[0]
        else:
            low_s = candidate_min_verts[-1]
        if len(candidate_max_verts) == 0:
            high_s = xs[-1]
        else:
            high_s = candidate_max_verts[0]
        if low_s > high_s:
            low_s, high_s = high_s, low_s
        return float(low_s), float(high_s)

    # nearest
    low_s = xs[np_argmin(np_abs(xs - lower_bound))]
    high_s = xs[np_argmin(np_abs(xs - upper_bound))]
    if low_s > high_s:
        low_s, high_s = high_s, low_s
    return float(low_s), float(high_s)

def snap_bounds_to_mesh_nodes(mesh, 
                              test_domain: NDArray, 
                              mode: str):
    """
    test_domain: (NDArray): An NDArray defining the bounding box. Each row represents a dimension
                            (e.g., test_domain[0] for x, test_domain[1] for y, test_domain[2] for z, if domain is 3 dimensional) 
                            and each row has two columns containing the lower_bound and upper_bound of each dimension.
                            Only dims < mesh.geometry().dim() used.
                            Example: [[0,1],[0,0.5]]
    mode: 'lower'|'upper'|'nearest'
    Returns snapped_bounds with same keys for used dims.

    Uses mesh.vertex coordinates (mesh.coordinates()).
    """
    mesh_max_dim = mesh.geometry().dim()
    coords = mesh.coordinates()
    snapped = {}
    for curr_dim in range(test_domain.shape[0]):
        lower_bound = min(test_domain[curr_dim])
        upper_bound = max(test_domain[curr_dim])
        if curr_dim >= mesh_max_dim:
            continue
        low_s, high_s = _snap_1d_coords_to_mesh_nodes(coords[:, curr_dim], lower_bound, upper_bound, mode)
        snapped[curr_dim] = (low_s, high_s)
    return snapped

def mark_cells_by_midpoint_bounds(mesh, 
                                  test_domain: NDArray):
    """
    Mark cell = 1 if its midpoint lies within the axis-aligned box defined by test_domain.
    test_domain (NDArray): An NDArray defining the bounding box. Each row represents a dimension
                            (e.g., test_domain[0] for x, test_domain[1] for y, test_domain[2] for z, if domain is 3 dimensional) 
                            and each row has two columns containing the lower_bound and upper_bound of each dimension.
                            Example: [[0,1],[0,0.5]]
    """
    mesh_max_dim = mesh.geometry().dim()
    cell_markers = MeshFunction("size_t", mesh, mesh_max_dim)
    cell_markers.set_all(0)

    for cell in cells(mesh):
        mid = cell.midpoint().array()
        midpoint_flag = True
        for curr_dim in range(test_domain.shape[0]):
            lower_bound = min(test_domain[curr_dim])
            upper_bound = max(test_domain[curr_dim])
            if curr_dim < mesh_max_dim and not (lower_bound <= mid[curr_dim] <= upper_bound):
                midpoint_flag = False
                break
        if midpoint_flag:
            cell_markers[cell] = 1
    return cell_markers

def get_cell_markers_from_test_domain(mesh, 
                               test_domain: NDArray, 
                               cell_marker_policy: str = 'all'):
    """
    Creates a FEniCS CellFunction by marking cells based on whether their
    vertices lie within a specified bounding box.

    This provides a 'node-aligned' region.

    Args:
        mesh (Mesh): The FEniCS mesh object.
        test_domain (NDArray): An NDArray defining the bounding box. Each row represents a dimension
                               (e.g., test_domain[0] for x, test_domain[1] for y, test_domain[2] for z, if domain is 3 dimensional) 
                               and each row has two columns containing the lower_bound and upper_bound of each dimension.
                               Example: [[0,1],[0,0.5]]
        cell_marker_policy (str): The rule for including a cell:
                      - 'all': Mark the cell if ALL of its vertices are inside
                               the test_domain. This creates an "inner bound" region.
                      - 'any': Mark the cell if AT LEAST ONE of its vertices
                               is inside the test_domain. This creates an "outer bound" region.
    Returns:
        MeshFunction: A cell function where cells satisfying the cell_marker_policy are
                      marked with 1, and all others are marked with 0.
    """
    # ensure the cell_marker_policy is valid
    if cell_marker_policy not in ['all', 'any']:
        raise ValueError("Policy must be either 'all' or 'any'.")
    
    # get mesh dimension and geometry data
    max_mesh_dim = mesh.geometry().dim()
    all_vertex_coords = mesh.coordinates()
    cell_to_vertex_map = mesh.cells()
    
    # initialize a MeshFunction to store the integer markers for each cell.
    # the second argument is the topological dimension of the entity to be marked (cells).
    cell_markers = MeshFunction("size_t", mesh, max_mesh_dim)
    # default all cells to marker 0 (outside)
    cell_markers.set_all(0)

    # iterate through each cell by its index, 'cell_idx'
    for cell_idx in range(mesh.num_cells()):
        # get the indices of the vertices that make up this cell
        vertex_indices_for_cell = cell_to_vertex_map[cell_idx]
        # get the actual coordinates of these vertices
        coords_of_cell_vertices = all_vertex_coords[vertex_indices_for_cell]

        #perform the test_domain check for all vertices in the cell at once:
        # start by assuming all vertices are inside
        is_inside_flags = np_ones(len(vertex_indices_for_cell), dtype=bool)
        # sequentially apply the filter for each dimension in the test_domain
        for curr_dim in range(test_domain.shape[0]):
            lower_bound = min(test_domain[curr_dim])
            upper_bound = max(test_domain[curr_dim])
            # skip if the dimension is not relevant to the mesh
            if curr_dim >= max_mesh_dim:
                continue
            # get the coordinates for the current dimension for all vertices
            vertex_coords_in_dim = coords_of_cell_vertices[:, curr_dim]     
            # update the flags using a vectorized boolean AND operation.
            # a vertex is only 'still inside' if it was inside before AND it's within the test_domain for the current dimension.
            is_inside_flags &= (vertex_coords_in_dim >= lower_bound) & (vertex_coords_in_dim <= upper_bound)
            
        #apply the chosen cell_marker_policy      
        if cell_marker_policy == 'all':
            # np_all() returns True only if every element in the array is True.
            if np_all(is_inside_flags):
                cell_markers[cell_idx] = 1              
        elif cell_marker_policy == 'any':
            # np_any() returns True if at least one element in the array is True.
            if np_any(is_inside_flags):
                cell_markers[cell_idx] = 1
                
    return cell_markers

def inject_cell_markers_from_numpy_array_to_MeshFunction(mesh, cell_marker_numpy_array_extract: NDArray):
    dim = mesh.topology().dim()
    cell_markers = MeshFunction("size_t", mesh, dim)
    cell_markers.array()[:] = cell_marker_numpy_array_extract
    return cell_markers

def assemble_mass_matrix_from_mesh(V: FunctionSpace, 
                                   cell_markers: MeshFunctionSizet = None):
    """
    assemble consistent mass matrix over whole domain (marker None) or subset of domain.
    Returns a PETSc.Mat object of the mass matrix.
    """
    dx = df_Measure("dx", domain=V.mesh(), subdomain_data=cell_markers)
    sub_id = 1 if cell_markers is not None else 0
    f_trial = df_TrialFunction(V)
    f_test = df_TestFunction(V)
    mass_matrix = df_assemble(df_inner(f_trial, f_test) * dx(sub_id))

    return df_as_backend_type(mass_matrix).mat()


def generate_xdmf_from_dot_msh(mesh_save_dir, mesh_save_name):
    """..."""
    #----------- error check(s) -----------
    #--------------------------------------

    mesh = meshio.read(f"{mesh_save_dir}/{mesh_save_name}.msh")
    cells = {"triangle": mesh.cells_dict["triangle"]}

    #create new mesh with only the triangle elements
    triangle_mesh = meshio.Mesh(points=mesh.points[:,:2],
                            cells=cells,
                            cell_data={"triangle": [mesh.cell_data_dict["gmsh:geometrical"]["triangle"]]})
    
    #write xdmf using only the triangle mesh elements
    triangle_mesh.write(f"{mesh_save_dir}/{mesh_save_name}.xdmf")

def generate_1D_mesh(domain_interval: list = [0,1], 
                     mesh_num_of_steps: int = 128, 
                     mesh_save_dir: str = 'data/mesh_data', 
                     mesh_save_name: str = 'interval_mesh',
                     gen_xdmf: bool = True,
                     return_mesh: bool = False,
                     verbose=True):
    
    if not os_path.exists(mesh_save_dir):
        os_makedirs(mesh_save_dir, exist_ok=True)

    mesh = df_IntervalMesh(mesh_num_of_steps, min(domain_interval), max(domain_interval))

    if gen_xdmf:
        file_name_and_path = f'{mesh_save_dir}/{mesh_save_name}.xdmf'
        with df_XDMFFile(file_name_and_path) as xdmf:
            xdmf.write(mesh)
    if verbose:
        mesh = df_Mesh()
        with df_XDMFFile(file_name_and_path) as xdmf:
            xdmf.read(mesh)
            coords = mesh.coordinates()
            print("Mesh created successfully!")
            print(f"Number of cells:    {mesh.num_cells()}")
            print(f"Number of vertices: {mesh.num_vertices()}")
            print(f"Domain bounds:      [{coords.min()}, {coords.max()}]")
    if return_mesh:
        return mesh
    

def generate_2D_mesh(domain_height, domain_width, mesh_steps, mesh_save_dir, mesh_save_name='rectangle', gen_xdmf=True):
    """..."""
    #----------- error check(s) -----------
    #--------------------------------------

    gmsh.initialize()
    gmsh.clear()

    gmsh.model.add("domain_mesh")

    points = [
        gmsh.model.geo.addPoint(0, 0, 0, mesh_steps),
        gmsh.model.geo.addPoint(domain_width, 0, 0, mesh_steps),
        gmsh.model.geo.addPoint(domain_width, domain_height, 0, mesh_steps),
        gmsh.model.geo.addPoint(0, domain_height, 0, mesh_steps)
    ]

    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0])
    ]

    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    #save .msh
    gmsh.write(f"{mesh_save_dir}/{mesh_save_name}.msh")
    gmsh.finalize()

    if gen_xdmf:
        generate_xdmf_from_dot_msh(mesh_save_dir=mesh_save_dir, mesh_save_name=mesh_save_name)