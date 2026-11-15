################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import os 
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import ast

from tools.io_utils import writeOBJ, writeTriangularSurfaceMesh, \
                            readTriangularSurfaceMesh
                            
from tools.pointclouds import createCoordinateSystem, createRightHandSystem, \
                                get_projected_area, computeFaceArea, \
                                get_ratio_from_bnd, get_height_from_bnd, \
                                get_circularity, getOrderedBoundaryCurve, \
                                get_circumference, alignCoordinateSystems, \
                                get_width_from_bnd



def compute_accurate_3d_orifice_area(v, faces_ant, faces_post):
    '''
    Compute the exact three-dimensional orifice area of the mitral valve
    (using Juliana's algorithm). 

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    faces_ant : (P, dim) numpy nd array
        Faces of anterior leaflet.
    faces_post : (Q, dim) numpy nd array
        Faces of posterior leaflet.

    Returns
    -------
    A_3d_orifice : float
        3d orifice area of mitral valve.

    '''
    
    # write path to file in specific path folder
    file_ant = "pymeshlab/shape_anterior.stl"
    file_post = "pymeshlab/shape_posterior.stl"
     
    writeTriangularSurfaceMesh(v, faces_ant, file_ant)
    writeTriangularSurfaceMesh(v, faces_post, file_post)
    
    # execute pymeshlab file to find self intersections
    try:
        output = subprocess.getoutput("python pyora3d.py")
        # transform output
        A_3d_orifice = ast.literal_eval(output)
    except:
        A_3d_orifice = 0.0
        print("- PyVista Error")
        

    return A_3d_orifice



def read_scalars_from_obj(mesh_path):
    '''
    Read first line encrypted as a comment, contains scalar information of the shape/mesh.

    Parameters
    ----------
    mesh_path : string
        Absolute path to the .OBJ mesh file to analyze.

    Returns
    -------
    scalar_array : (N,) numpy array
        Array of N elements, scalar properties of the OBJ file.

    '''
    
    with open(mesh_path) as f:
        first_line = f.readline().strip('\n')
    
    # remove all string characters
    temp = first_line.replace("(", "")
    temp = temp.replace(")", "")
    temp = temp.replace("'", "")
    temp = temp.replace("#", "")
    
    # read string as array of floats
    scalar_array = np.fromstring(temp, dtype=float, sep=',')
    
    return scalar_array



def align_to_commissure(v, f, annulus, lat_com, med_com, v_hori, mesh_path, save_geo=False):   
    '''
    Align mitral valve geometry along commissure points through vertex id's of TomTec data and normal vector of
    point cloud plane.
    Saves new geometry in same folder with same mesh name, with added "_aligned"

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    f : (M, dim) numpy nd array
        definition of triangles via vertex indices.
    annulus : (P, dim) numpy nd array
        vertices of annulus, first boundary of mitral valve coordinates.
    lat_com : integer
        vertex id of lateral commissure point.
    med_com : integer
        vertex id of medial commissure point.
    v_hori : integer
        vertex id of horizontal reference point on annulus.
    mesh_path : string
        absolute string path of mesh.
    save_geo : bool, optional
        flag to save the aligned geometry. The default is False.
        
    Returns
    -------
    v_com_plane : (N,dim) numpy nd array
        vertices of aligned geometry.
        
    '''
    # ubuntu or windows dependent delimiter
    # check if running on windows 
    if os.name == "nt":
        # windows
        delimiter = "\\"
    else:
        # linux
        delimiter = "/"
        
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    [a1, a2, a3], ctr_annulus = createCoordinateSystem(np.array(annulus))

    # realign the vertices to have equal orientation for all possible original orientations
    # check if geometry is already aligned, so that normal vector (a3) of SVD is already equal to [0,0,1] base vector
    e1 = v[med_com, :] - v[lat_com, :]
    e2 = v[v_hori, :] - ctr_annulus
    e3 = np.cross(e1, e2)

    ctr = v.mean(axis=0)
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]
    v_com = v[lat_com, :] - v[med_com, :]
    
    # align
    if np.dot(e3, a3) > 0:
        # use negative base vector for orientation
        base_com = createRightHandSystem(v_com, -a3)    

    else:
        # use positive base vector for orientation
        base_com = createRightHandSystem(v_com, a3)        
    
    v_com_plane, _, _, _ = alignCoordinateSystems(v, ctr, base_com, origin, base_target)

    
    # save mesh transformed into annulus plane and oriented along commissure points
    if save_geo == True:
        path = ""
        for i in range(len(mesh_path.split(delimiter))-1):
            path = path + os.path.join("/", mesh_path.split(delimiter)[i])
        file_aligned = os.path.join(path, mesh_path.split(delimiter)[-1][:-4] + "_aligned.obj")    # .obj / .stl
        
        writeOBJ(file_aligned, v_com_plane, f)
    
    return v_com_plane


def get_multiple_face_area(v, f):
    '''
    Compute the surface area of the specific faces.

    Parameters
    ----------
    v : (N,dim) numpy nd array
        Vertices of .stl/.obj file of mitral valve.
    f : (M, dim) numpy nd array
        Definition of triangles via vertex indices of the faces to compute area.

    Returns
    -------
    A : float
        Surface area in mm^2
    
    '''
    # surface area of leaflet
    f = np.array(f, dtype=int)
    A = 0
    for i in range(len(f)):
        A = A + computeFaceArea(v[f[i, 0], :], v[f[i, 1], :], v[f[i, 2], :])

    return A


def get_leaflet_length_path(v, annulus, lat_com, med_com, id_annu, id_orif, nr_v_circ, nr_layers, axis_dim):
    '''
    Compute the leaflet length of the mitral valve
    MV aligned by commissures and centered via ctr of mass
    Leaflet length is distance between two boundary points of leaflet
        - one on annulus bnd curve
        - and path on the grid line 
    Picked as points closest to y-axis (axis intersecting the leaflets)

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    annulus : (P, dim) numpy nd array
        vertices of annulus, first boundary of mitral valve coordinates.
    lat_com : integer
        vertex id of lateral commissure point.
    med_com : integer
        vertex id of medial commissure point.
    id_annu : (N,) numpy array
        vertex id's annulus boundary for one leaflet - only one half of circumference.
    id_orif : (N,) numpy array
        vertex id's orifice boundary for one leaflet - only one half of circumference.
    axis_dim : integer
        pick axis orthogonal to commissure axis - either x (0) or y (1).

    Returns
    -------
    len_lf : float
        length of leaflet
        
    '''
    # alignment of coordinate system - which dimension/axis for lenght calculation - axis_dim
    annulus = np.array(annulus)
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(annulus)

    ctr = annulus.mean(axis=0)
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]

    v_com = v[lat_com, :] - v[med_com, :]
    # use negative base vector for orientation
    base_com = createRightHandSystem(v_com, -a3)
    v_com_plane, _, _, _ = alignCoordinateSystems(v, ctr, base_com, origin, base_target)

    # calculate center of all vertices - translate to center
    ctr_new = v_com_plane.mean(axis=0)
    v_com_plane = v_com_plane - ctr_new
    
    # get vertices of one leaflet for annulus boundary curve
    v_annu = v_com_plane[id_annu, :]

    # minimum value of horizontal axis 
    min_annu_idx = np.where(abs(v_annu) == min(abs(v_annu[:, axis_dim])))[0]
    # index vertex of all vertices
    start_annu_idx = np.where(v_com_plane == v_annu[min_annu_idx])[0]
    
    # indices for path 
    len_lf_all = []
    for i in range(nr_layers-1):
        # indices of pieces
        idx0 = min_annu_idx - i     * nr_v_circ
        idx1 = min_annu_idx - (i+1) * nr_v_circ
        
        # piecewise length
        v_leaf_i = v_com_plane[idx1, :] - v_com_plane[idx0, :]

        # compute leaflet length
        len_lf_i = np.linalg.norm(v_leaf_i, 2)

        # add length piece to len_lf 
        len_lf_all.append(len_lf_i)
        
    # total length
    len_lf = sum(len_lf_all)
    
    return len_lf


def get_commissural_width_from_bnd(v, lat_com, med_com, nr_v_circ, nr_layers): 
    '''
    Calculate the commissural width. Defined as the extended path of the 
    commissure vertices. Due to the vertex structure of the MV TomTec meshes, 
    walking along a path from the commissure points to the annulus boundary 
    curve is possible.

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    lat_com : integer
        vertex id of lateral commissure point.
    med_com : integer
        vertex id of medial commissure point.
    nr_v_circ : integer
        Number of circular points in the MV TomTec mesh.
    nr_layers : integer
        Number of layers from orifice to annulus boundary.

    Returns
    -------
    cw : float
        Commissural width.

    '''
    # get vertex index of lateral outer boundary vertex
    idx_bnd_lat = lat_com - nr_v_circ*(nr_layers-1)
    # get vertex index of medial outer boundary vertex
    idx_bnd_med = med_com - nr_v_circ*(nr_layers-1)

    # piecewise length
    v_cw = v[idx_bnd_lat, :] - v[idx_bnd_med, :]
        
    # commissural width
    cw = np.linalg.norm(v_cw, 2)

    return cw


def get_leaflet_length_approx(v, annulus, lat_com, med_com, id_annu, id_orif, axis_dim):
    '''
    Compute the leaflet length of the mitral valve
    MV aligned by commissures and centered via ctr of mass
    Leaflet length is distance between two boundary points of leaflet
        - one on annulus bnd curve
        - one on orifice bnd curve
    Picked as points closest to y-axis (axis intersecting the leaflets)

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    annulus : (P, dim) numpy nd array
        vertices of annulus, first boundary of mitral valve coordinates.
    lat_com : integer
        vertex id of lateral commissure point.
    med_com : integer
        vertex id of medial commissure point.
    id_annu : (N,) numpy array
        vertex id's annulus boundary for one leaflet - only one half of circumference.
    id_orif : (N,) numpy array
        vertex id's orifice boundary for one leaflet - only one half of circumference.
    axis_dim : integer
        pick axis orthogonal to commissure axis - either x (0) or y (1).

    Returns
    -------
    len_lf : float
        length of leaflet
    '''
    # alignment of coordinate system - which dimension/axis for lenght calculation - axis_dim
    annulus = np.array(annulus)
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(annulus)

    ctr = annulus.mean(axis=0)
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]

    v_com = v[lat_com, :] - v[med_com, :]
    # use negative base vector for orientation
    base_com = createRightHandSystem(v_com, -a3)
    v_com_plane, _, _, _ = alignCoordinateSystems(v, ctr, base_com, origin, base_target)

    # calculate center of all vertices - translate to center
    ctr_new = v_com_plane.mean(axis=0)
    v_com_plane = v_com_plane - ctr_new
    
    # get vertices of one leaflet for annulus boundary curve and orifice curve
    v_orif = v_com_plane[id_orif, :]
    v_annu = v_com_plane[id_annu, :]

    # minimum value of horizontal axis 
    min_orif_idx = np.where(abs(v_orif) == min(abs(v_orif[:, axis_dim])))[0]
    min_annu_idx = np.where(abs(v_annu) == min(abs(v_annu[:, axis_dim])))[0]
    v_leaf = v_orif[min_orif_idx, :] - v_annu[min_annu_idx, :]

    # compute leaflet length
    len_lf = np.linalg.norm(v_leaf,2)

    return len_lf


def get_leaflet_angle(v, annulus, lat_com, med_com, id_annu, id_orif, axis_dim):
    '''
    Compute the leaflet angle of the mitral valve
    MV aligned by commissures and centered via ctr of mass
    Leaflet length is distance between two boundary points of leaflet
        - one on annulus bnd curve
        - one on orifice bnd curve
    Picked as points closest to y-axis (axis intersecting the leaflets)

    Parameters
    ----------
    v : (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    annulus : (P, dim) numpy nd array
        vertices of annulus, first boundary of mitral valve coordinates.
    lat_com : integer
        vertex id of lateral commissure point.
    med_com : integer
        vertex id of medial commissure point.
    id_annu : (N,) numpy array
        vertex id's annulus boundary for one leaflet - only one half of circumference.
    id_orif : (N,) numpy array
        vertex id's orifice boundary for one leaflet - only one half of circumference.
    axis_dim : integer
        pick axis orthogonal to commissure axis - either x (0) or y (1).

    Returns
    -------
    ang_lf : float
        leaflet angle
    '''
    # alignment of coordinate system - which dimension/axis for lenght calculation - axis_dim
    annulus = np.array(annulus)
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(annulus)

    v_com_plane = v
    # calculate center of all vertices - translate to center
    ctr_new = v_com_plane.mean(axis=0)
    v_com_plane = v_com_plane - ctr_new
    
    # get vertices of one leaflet for annulus boundary curve and orifice curve
    v_orif = v_com_plane[id_orif, :]
    v_annu = v_com_plane[id_annu, :]

    # minimum value of horizontal axis 
    min_orif_idx = np.where(abs(v_orif) == min(abs(v_orif[:, axis_dim])))[0]
    min_annu_idx = np.where(abs(v_annu) == min(abs(v_annu[:, axis_dim])))[0]
    v_leaf = v_orif[min_orif_idx, :] - v_annu[min_annu_idx, :]

    # compute leaflet angle anterior: between length anterior line and annulus plane a3
    unit_leaf = v_leaf / np.linalg.norm(v_leaf)
    unit_a3 = a3 / np.linalg.norm(a3)
    ang_lf = ( 90 - np.arccos(np.dot(unit_leaf, unit_a3)) / np.pi * 180 )[0]
        
    return ang_lf


def write_leaflet_stl(v, faces_ant, faces_post, mesh_path, delimiter):
    '''
    Generate leaflet .stl files for both leaflets: anterior and posterior.

    Parameters
    ----------
    v : (N,dim) numpy nd array
        Vertices of .stl/.obj file of mitral valve.
    faces_ant : (P, dim) numpy nd array
        Faces of anterior leaflet.
    faces_post : (Q, dim) numpy nd array
        Faces of posterior leaflet.
    mesh_path : string
        Absolute string path of mesh.
    delimiter: string
        Delimiter of paths, dependent on OS system.

    Returns
    -------
    None.
    
    '''
    file_lf_ant = mesh_path.split(delimiter)[-1][:-4] + "_anterior.stl"    # .obj / .stl
    file_lf_post = mesh_path.split(delimiter)[-1][:-4] + "_posterior.stl"    # .obj / .stl
    
    save_path = mesh_path.rsplit(delimiter, 2)[0]
    folder = mesh_path.rsplit(delimiter, 2)[1] + "_leaflets"
    
    save_path = os.path.abspath(save_path + "/" + folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    lf_ant_path = os.path.join(save_path, file_lf_ant)
    lf_post_path = os.path.join(save_path, file_lf_post)

    writeTriangularSurfaceMesh(v, faces_ant, lf_ant_path)
    writeTriangularSurfaceMesh(v, faces_post, lf_post_path)
    


def get_3d_orifice_area(v, v_orif, id_orif, lat_com, med_com, nr_v, mesh_path, delimiter):
    '''
    Compute an broad approximation of the 3-dimensional orifice area.
    The approximation closes the orifice edges with triangles and computes its
    surface area. 

    Parameters
    ----------
    v : (N,dim) numpy nd array
        Vertices of .stl/.obj file of mitral valve.
    v : (M,dim) numpy nd array
        Vertices of of the orifice boundary of the mitral valve.
    id_orif : (N,) numpy array
        Vertex id's orifice boundary.
    lat_com : integer
        Vertex id of lateral commissure point.
    med_com : integer
        Vertex id of medial commissure point.
    nr_v : integer
        Number of vertices in one circumference layer.
    mesh_path : string
        Absolute string path of mesh.
    delimiter: string
        Delimiter of paths, dependent on OS system.

    Returns
    -------
    A : float
        Approximated 3-dimensional surface area of the orifice.

    '''
    lat_com = 1580
    med_com = 1540
    
    # number of triangles
    nr_tri = nr_v - 2
    
    # initialization of face matrix
    f_orif = np.zeros((nr_tri, 3))
    
    # medial and lateral ending triangle
    f_orif[0,:] = [med_com+1, med_com, med_com-1]
    f_orif[1,:] = [lat_com+1, lat_com, lat_com-1]
    
    # loop over left triangles
    k = 2
    for i in range(1, int((nr_tri-2)/2)+1):
        if 1519 < med_com - i < lat_com:
            f_orif[k,:] = [med_com + i + 1, med_com + i, med_com - i ]
        else: 
            f_orif[k,:] = [med_com + i + 1, med_com + i, med_com - i + nr_v ]
        
        k += 1
        
    # loop over right triangles
    for i in range(1, int((nr_tri-2)/2)+1):
        if 1520 < med_com - i < lat_com:
            f_orif[k,:] = [med_com + i + 1, med_com - i, med_com - i - 1 ]
        elif med_com - i == 1520: 
            f_orif[k,:] = [med_com + i + 1, med_com - i, med_com + nr_v - i - 1 ]
        else:
            f_orif[k,:] = [med_com + i + 1, med_com + nr_v - i , med_com + nr_v - i - 1 ]
        
        k += 1
        

    # save geometry
    orifice_file = mesh_path.split(delimiter)[-1][:-4] + "_orifice.stl"
    
    save_path = mesh_path.rsplit(delimiter, 2)[0]
    folder = mesh_path.rsplit(delimiter, 2)[1] + "_orifice"
    save_path = os.path.abspath(save_path + "/" + folder)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    orifice_path = os.path.join(save_path, orifice_file)
    
    writeTriangularSurfaceMesh(v, f_orif, orifice_path)
      
    A = get_multiple_face_area(v, f_orif)
    
    return A


def getMetricsMain(mesh_path, work_path, use_metrics, parameters):
    '''
    Computes selected metrics of the mitral valve.

    Parameters
    ----------
    mesh_path : string
        absolute path of the mesh to analyze.
    work_path : string
        absolute path to the working directory.
    use_metrics : dictionary
        contains all available metrics with value 0 or 1.
        0 : not included
        1 : metric is computed
    parameters : dict
        dictionary containing all geometry specific input parameters from infile.py.

    Returns
    -------
    metrics : dict
        Dictionary of metrics.
        
    '''   
    # path for face files for anterior and posterior leaflets
    file_ant = os.path.join(work_path, "faces_ant.txt")
    file_post = os.path.join(work_path, "faces_post.txt")
    
    # determined vertex id's
    # depends on definition of vertex id structure
    lat_com = parameters["lat_com"]
    med_com = parameters["med_com"]
    v_hori = parameters["v_hori"]
    
    nr_v_circ = parameters["nr_v_circ"]
    nr_layers = parameters["nr_layers"]
    
    bnd_id_annulus = parameters["bnd_id_annulus"]
    bnd_id_orifice = parameters["bnd_id_orifice"]
    
    # vertex id's on boundaries of leaflets
    id_annu_ant = parameters["id_annu_ant"]
    id_orif_ant = parameters["id_orif_ant"]
    id_annu_post = parameters["id_annu_post"]
    id_orif_post = parameters["id_orif_post"]
    id_orif = parameters["id_orif"]
    
    # ------------------------------------------------------------------------------------------
    # ubuntu or windows dependent delimiter
    # check if running on windows 
    if os.name == "nt":
        # windows
        delimiter = "\\"
    else:
        # linux
        delimiter = "/"
    # ------------------------------------------------------------------------------------------
    # read mesh
    v, f = readTriangularSurfaceMesh(mesh_path)
    # get center
    ctr = v.mean(axis=0)
    # get boundary curve, check number of boundaries
    v_bn, f_bn = getOrderedBoundaryCurve(v, f, 2)
    # here : assumption that annulus is always first list item, orifice second item
    annulus = v_bn[bnd_id_annulus]
    orifice = v_bn[bnd_id_orifice]

    # get leaflet faces
    faces_ant = np.loadtxt(file_ant)
    faces_post = np.loadtxt(file_post)
    # start counting from zero
    faces_ant = faces_ant - 1
    faces_post = faces_post - 1
    # ------------------------------------------------------------------------------------------
    # save aligned geometry
    # use v_com_plane as input for all functions, so same orientation garantueed
    v_com_plane = align_to_commissure(v, f, annulus, lat_com, med_com, v_hori, mesh_path, save_geo=False)

    # get boundary curve of aligned geometry for eventual realginment
    v_bn_com, f_bn_com = getOrderedBoundaryCurve(v_com_plane, f, 2)
    annulus_com = v_bn_com[bnd_id_annulus]
    orifice_com = v_bn_com[bnd_id_orifice]

    # save anterior and posterior leaflet
    write_leaflet_stl(v, faces_ant, faces_post, mesh_path, delimiter)

    # ------------------------------------------------------------------------------------------
    # get all activated mitral valve metrics
    # get circumference annulus
    if "circ_annulus" in use_metrics:
        circ_annulus = get_circumference(annulus)

    # get circumference orifice
    if "circ_orifice" in use_metrics:
        circ_orifice = get_circumference(orifice)

    # get annulus area
    if "A_annulus" in use_metrics:
        A_annulus = get_projected_area(annulus)

    # get orifice area
    if "A_orifice_orif" in use_metrics:
        A_orifice_orif = get_projected_area(orifice)

    # get annulus height
    if "h_annulus" in use_metrics:
        h_annulus = get_height_from_bnd(annulus_com)

    # get orifice height
    if "h_orifice" in use_metrics:
        h_orifice = get_height_from_bnd(orifice_com)

    # get annulus ratio
    if "ratio_annulus" in use_metrics:
        ratio_annulus = get_ratio_from_bnd(v, annulus, lat_com, med_com)

    # get orifice ratio
    if "ratio_orifice" in use_metrics:
        ratio_orifice = get_ratio_from_bnd(v, orifice, lat_com, med_com)

    # get annulus circularity
    if "ci_annulus" in use_metrics:
        circ_annulus = get_circumference(annulus)
        A_annulus = get_projected_area(annulus)
        ci_annulus = get_circularity(A_annulus, circ_annulus)

    # get annulus circularity
    if "ci_orifice" in use_metrics:
        circ_orifice = get_circumference(orifice)
        A_orifice_orif = get_projected_area(orifice)
        ci_orifice = get_circularity(A_orifice_orif, circ_orifice)

    # get anterior leaflet area
    if "A_lf_ant" in use_metrics:
        A_lf_ant = get_multiple_face_area(v, faces_ant)

    # get anterior leaflet area
    if "A_lf_post" in use_metrics:
        A_lf_post = get_multiple_face_area(v, faces_post)

    # get anterior leaflet length
    if "len_lf_ant" in use_metrics:
        len_lf_ant = get_leaflet_length_approx(v_com_plane, annulus_com, lat_com, \
                                        med_com, id_annu_ant, id_orif_ant,  axis_dim=0)
        # len_lf_ant = get_leaflet_length_path(v_com_plane, annulus_com, lat_com, \
        #                                 med_com, id_annu_ant, id_orif_ant, nr_v_circ, nr_layers, axis_dim=0)
    
    # get posterior leaflet length
    if "len_lf_post" in use_metrics:
        len_lf_post = get_leaflet_length_approx(v_com_plane, annulus_com, lat_com, \
                                          med_com, id_annu_post, id_orif_post, axis_dim=0)
        # len_lf_post = get_leaflet_length_path(v_com_plane, annulus_com, lat_com, \
        #                                 med_com, id_annu_post, id_orif_post, nr_v_circ, nr_layers, axis_dim=0)
        
    # get anterior angle of leaflet
    if "ang_lf_ant" in use_metrics:
       ang_lf_ant = get_leaflet_angle(v_com_plane, annulus_com, lat_com, \
                                      med_com, id_annu_ant, id_orif_ant, axis_dim=0)

    # get posterior angle of leaflet
    if "ang_lf_post" in use_metrics:
        ang_lf_post = get_leaflet_angle(v_com_plane, annulus_com, lat_com, \
                                        med_com, id_annu_post, id_orif_post, axis_dim=0)

    # get approximated 3d orifice area
    if "A_3d_orifice_approx" in use_metrics:
        A_3d_orifice_approx = get_3d_orifice_area(v_com_plane, orifice_com, id_orif, 
                                           lat_com, med_com, nr_v_circ, mesh_path, delimiter)

    # get exact 3d orifice area
    if "A_3d_orifice" in use_metrics:
        A_3d_orifice = compute_accurate_3d_orifice_area(v_com_plane, faces_ant, faces_post)
        
    # get cardiac surgery metric : annulus height to width (in commissure direction)
    if "height_to_width" in use_metrics:
        h_annulus = get_height_from_bnd(annulus_com)
        # w_annulus = get_width_from_bnd(v, annulus, lat_com, med_com)
        cw = get_commissural_width_from_bnd(v, lat_com, med_com, nr_v_circ, nr_layers)
        height_to_width = h_annulus / cw


    # write all computed metrics into metrics dictionary
    metrics = {}
    for count, key in enumerate(use_metrics.keys()):
        metrics[key] = locals()[key]

    return metrics