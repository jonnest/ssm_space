################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkCommonDataModel import ( vtkCellArray, 
                                        	vtkIterativeClosestPointTransform, 
                                            vtkPolyData )
                                            
from tools.io_utils import readTriangularSurfaceMesh, writeTriangularSurfaceMesh


def createRightHandSystem(e1, e3):
    '''
    Create a right hand system from two vectors.
    e3 will not be modified, e1 will be projected to be orthogonal to e3.
    e2 is computed via cross product.

    Parameters
    ----------
    e1 : (3,) numpy array
        arbitrary vector, projected as y-axis for future right-hand-coordinate-system
    e3 : (3,) numpy array
        z vector of future right-hand-coordinate-system

    Returns
    -------
    base : (3,3) numpy nd array
        b1, b2, b3 as orthogonal bases of future right-hand-coordinate-system (b_i columns in matrix)
    '''
    
    e1 = e1 - np.dot(e1, e3)*e3
    e1 = e1/np.linalg.norm(e1)
    e2 = np.cross(e3,e1)

    # negative sign for all base vectors for defining coordinate system specifically
    base = [ e1/np.linalg.norm(e1,2), e2/np.linalg.norm(e2,2), e3/np.linalg.norm(e3,2)]

    return base
    
    
def get_circumference(bnd_array):
    '''
    Compute the circumference of a closed line of vertices

    Parameters
    ----------
    bnd_array : (dim, N) numpy nd array
        boundary array containing coordinates of a closed boundary

    Returns
    -------
    circ_bnd : float
        circumference as sum of all distances between two consecutive vertices
        (unit corresponds to the unit of the vertex coordinates)
    '''
    # loop
    circ_bnd = 0
    for i in range(len(bnd_array)-1):
        v_temp = bnd_array[i] - bnd_array[i+1]
        l_temp = np.linalg.norm(v_temp,2)
        circ_bnd = circ_bnd + l_temp
    # for periodic condition
    v_temp = bnd_array[i+1] - bnd_array[0]
    l_temp = np.linalg.norm(v_temp, 2)
    circ_bnd = circ_bnd + l_temp
    
    return circ_bnd


def get_projected_area(bnd_array):
    '''
    Compute the projected area of a 3d closed loop of vertices.
    Projected area to the best fitting plane (a1,a2 of SVD) in the point cloud.

    Parameters
    ----------
    bnd_array : (dim, N) numpy nd array
        boundary array containing coordinates of a closed boundary

    Returns
    -------
    A_bnd : float
        projected area (unit corresponds to squared the unit of the vertex coordinates)
    '''
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    bnd_array = np.array(bnd_array)
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(bnd_array)
    # transform to xy plane
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]
    base_bnd = [a1, a2, a3]
    # transform bnd to xy plane of bnd
    bnd_plane, _, _, _ = alignCoordinateSystems(bnd_array, ctr_bnd, base_bnd.copy(), origin, base_target.copy())

    bnd_plane[:,2] = 0
    A_bnd = poly_area(bnd_plane, base_target[:,-1])

    return A_bnd


def get_height_from_bnd(bnd_array):
    '''
    Compute the height difference regarding the minimum and maximum expanse of the best fitting plane.

    Parameters
    ----------
    bnd_array : (dim, N) numpy nd array
        boundary array containing coordinates of a closed boundary

    Returns
    -------
    h_bnd : float
        height difference of pointcloud (unit corresponds to squared the unit of the vertex coordinates)
    '''
    # best fit plane of annulus using SVD of point cloud - rotate into plane
    bnd_array = np.array(bnd_array)
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(bnd_array)
    # transform to xy plane
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]
    base_bnd = [a1, a2, a3]
    ###base_bnd = createRightHandSystem(a1, a3)
    # transform bnd to xy plane of bnd
    bnd_plane, _, _, _ = alignCoordinateSystems(bnd_array, ctr_bnd, base_bnd, origin, base_target)
    h_bnd = max(bnd_array[:, 2]) - min(bnd_array[:, 2])

    '''
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.scatter3D(bnd_plane[:, 0], bnd_plane[:, 1], bnd_plane[:, 2], marker='o', alpha=0.5)

    # preparing vector
    t = np.linspace(-20, 20, 100)
    # e10 = a3[0]*t
    # e11 = a3[1]*t
    # e12 = a3[2]*t
    # ax.plot3D(e10, e11, e12, 'red')
    base_com = base_bnd.copy()
    v10 = base_com[0][0] / np.linalg.norm(base_com[0]) * t
    v11 = base_com[0][1] / np.linalg.norm(base_com[0]) * t
    v12 = base_com[0][2] / np.linalg.norm(base_com[0]) * t
    ax.plot3D(v10, v11, v12, 'orange')

    v20 = base_com[1][0] / np.linalg.norm(base_com[1]) * t
    v21 = base_com[1][1] / np.linalg.norm(base_com[1]) * t
    v22 = base_com[1][2] / np.linalg.norm(base_com[1]) * t
    ax.plot3D(v20, v21, v22, 'green')

    v30 = base_com[2][0] / np.linalg.norm(base_com[2]) * t
    v31 = base_com[2][1] / np.linalg.norm(base_com[2]) * t
    v32 = base_com[2][2] / np.linalg.norm(base_com[2]) * t
    ax.plot3D(v30, v31, v32, 'red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    '''
    return h_bnd


def get_width_from_bnd(v, bnd_array, lat_com, med_com): 
    '''
    Compute the width of a closed loop pointcloud after aligned the mitral valve along the commissure points.
    y-axis is parallel to vertex between the commissure points, z-axis is orthogonal to best fitting plane.
    Compute as : x-axis bbox length.

    Parameters
    ----------
    v: (N,dim) numpy nd array
        Vertices of .stl/.obj file of mitral valve.
    bnd_array : (dim, N) numpy nd array
        Boundary array containing coordinates of a closed boundary.
    lat_com : integer
        Vertex id of lateral commissure point.
    med_com : integer
        Vertex id of medial commissure point.

    Returns
    -------
    width_bnd : float
        Width of boundary pointcloud.
    '''
    # best fit plane of boundary using SVD of point cloud - rotate into plane
    bnd_array = np.array(bnd_array)
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(bnd_array)

    ctr = bnd_array.mean(axis=0)
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]

    v_com = v[lat_com, :] - v[med_com, :]
    # use negative base vector for orientation
    base_com = createRightHandSystem(v_com, -a3)
    bnd_com_plane, _, _, _ = alignCoordinateSystems(bnd_array, ctr, base_com, origin, base_target)
    # bnd width
    # x axis dimension - bounding box of rotated MV for commissure alignment
    width_bnd = max(bnd_com_plane[:,0]) - min(bnd_com_plane[:,0])

    return width_bnd



def get_ratio_from_bnd(v, bnd_array, lat_com, med_com):
    '''
    Compute the axis ratio of a closed loop pointcloud after aligned the mitral valve along the commissure points.
    y-axis is parallel to vertex between the commissure points, z-axis is orthogonal to best fitting plane.
    Compute as : y-axis bbox length / x-axis bbox length.

    Parameters
    ----------
    v: (N,dim) numpy nd array
        vertices of .stl/.obj file of mitral valve.
    bnd_array : (dim, N) numpy nd array
        boundary array containing coordinates of a closed boundary
    lat_com : integer
        vertex id of lateral commissure point
    med_com : integer
        vertex id of medial commissure point

    Returns
    -------
    ratio_bnd : float
        ratio of boundary pointcloud
    '''
    # best fit plane of boundary using SVD of point cloud - rotate into plane
    bnd_array = np.array(bnd_array)
    [a1, a2, a3], ctr_bnd = createCoordinateSystem(bnd_array)

    ctr = bnd_array.mean(axis=0)
    base_target = np.identity(3)
    origin = [0.0, 0.0, 0.0]

    v_com = v[lat_com, :] - v[med_com, :]
    # use negative base vector for orientation
    base_com = createRightHandSystem(v_com, -a3)
    bnd_com_plane, _, _, _ = alignCoordinateSystems(bnd_array, ctr, base_com, origin, base_target)
    # bnd aspect ratio
    # long/short axis - bounding box of rotated MV for commissure alignment
    ratio_bnd = (max(bnd_com_plane[:,1]) - min(bnd_com_plane[:,1]))/    \
                (max(bnd_com_plane[:,0]) - min(bnd_com_plane[:,0]))

    return ratio_bnd


def get_circularity(A_bnd, circ_bnd):
    # circularity index bnd: 4*Pi*A/perimeter**2
    ci_bnd = (4*np.pi*A_bnd)/(circ_bnd**2)

    return ci_bnd



def poly_area(pts,normal):
    '''
    Computes area of in general convex 3D polygon.
    From https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates
    Points next to each other also need to be next to each other in array.

    Parameters
    ----------
    pts: (dim,N) numpy nd array
        coordinates of data points.
    normal: (dim,1) numpy nd array
        plane normal

    Returns
    -------
    area: float
        area of polygon
    '''
    pts = np.transpose(pts)
    if len(pts) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = pts.shape[1]
    for i in range(N):
        vi1 = pts[:,i]
        vi2 = pts[:,(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    area = abs(np.dot(total, normal)/2)
    return area


def computeFaceArea(a,b,c):
    '''
    Compute the surface area of a triangle defined by three vertices a, b and c.

    Parameters
    ----------
    a : (3,) numpy array
        coordinates of first vertex
    b : (3,) numpy array
        coordinates of second vertex
    c : (3,) numpy array
        coordinates of third vertex

    Returns
    -------
    A : float
        surface area of triangle (unit corresponds to squared the unit of the vertex coordinates)
    '''
    # compute face area via cross product
    ab = b - a
    ac = c - a
    v_cross = np.cross(ab, ac)
    A = np.linalg.norm(v_cross,2)/2
    return A



def alignCoordinateSystems(pts_org,ctr_org,base_org,ctr_target,base_target):
    '''
    Moves points of dimension dim with coordinate system and center to target base and center.
    First data is translated to target center, then dim-1 first base vectors are
    aligned to target base via rotation. Translation and Rotation tensors are returned
    in applied order. Only for 3D data.
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Parameters
    ----------
    pts: (N,dim) numpy nd array
        coordinates of data points.
    ctr: (dim,) numpy nd array
        origin of coordinate system.
    base: list of (dim,) numpy nd arrays of length dim
        base vectors of coordinate system calculated.
    ctr_target: (dim,) numpy nd array
        origin of target coordinate system.
    base_target: list of (dim,) numpy nd arrays of length dim
        base vectors of target coordinate system.

    Returns
    -------
    pts: (N,dim) numpy nd array
        coordinates of data points in target base.
    ctr: (dim,) numpy nd array
        origin of target coordinate system.
    base_new: list of (dim,) numpy nd arrays of length dim
        base vectors of target coordinate system.
    Movements: list of one (dim,) numpy nd array (translation) and dim-1 (dim,dim)
               numpy nd arrays (rotations)
        movements to project base on target base --> inverse application leads to original base
    '''
    # initialize new base to not change original 'base'
    base = base_org.copy()
    pts = pts_org.copy()
    ctr = ctr_org.copy()

    # Translation to target origin and store translation tensor
    Movements = [-(ctr-ctr_target)]
    pts = pts - (ctr-ctr_target) #np.expand_dims(ctr-ctr_target,axis=1)
    ctr = ctr_target
    # align base vectors via Rodriguez formula for rotation and store rotation tensor
    for i in range(len(base_target)-1):
        rot_axis = base[:-1]
        k = np.cross(rot_axis[i],base_target[i])
        k = k/(np.linalg.norm(k))
        K = np.array([[0,     -k[2], k[1]],
                      [k[2],  0,     -k[0]],
                      [-k[1], k[0],  0]])
        R = np.eye(3)+np.linalg.norm(np.cross(rot_axis[i],base_target[i]))*K + \
            (1-rot_axis[i].dot(base_target[i]))*K.dot(K)
        pts = pts.dot(R.T)
        ctr = R.dot(ctr)
        for j in range(len(base)):
            base[j] = R.dot(base[j])
        Movements.append(R)

    return pts,ctr,base,Movements



def iterative_closest_point(data_base, idx_target, data_out):
    '''
    Alignment via iterative closest point (IPC) algorithm using vtk modules
    (https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/) \n
    Algorithm following principal of rigid transformation
    This algorithm only works for grids with node correspondence
    give idx of target file of sorted items in list
       - best practice : calculate mean shape and use as target file

    Parameters
    ----------
    data_base : string
        absolute path of directory of data to align
    idx_target : integer
        index of target file in list of files in data_base dir
    data_out : string
        absolute path of directory to save aligned files

    Returns
    -------
    None.
    
    '''

    stl_files = glob.glob(data_base + "*.stl")
    stl_files = sorted(stl_files)
    print(stl_files)

    # read target data and set vtk variables for target
    v_tar, f_tar = readTriangularSurfaceMesh(stl_files[idx_target])

    targetPoints = vtkPoints()
    targetVertices = vtkCellArray()

    for i in range(len(v_tar)):
        tp_id = targetPoints.InsertNextPoint(v_tar[i, :])
        targetVertices.InsertNextCell(1)
        targetVertices.InsertCellPoint(tp_id)

    target = vtkPolyData()
    target.SetPoints(targetPoints)
    target.SetVerts(targetVertices)

    print("\nnumber of transformed files ", len(stl_files))
    print("\ntarget file ", stl_files[idx_target])

    for j in range(len(stl_files)):
        print("\nrun IPC algorithm for file nr ", j)

        v, f = readTriangularSurfaceMesh(stl_files[j])

        sourcePoints = vtkPoints()
        sourceVertices = vtkCellArray()

        # read source file
        for i in range(len(v)):
            sp_id = sourcePoints.InsertNextPoint(v[i, :])
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(sp_id)

        source = vtkPolyData()
        source.SetPoints(sourcePoints)
        source.SetVerts(sourceVertices)

        # ============ run ICP ==============
        icp = vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations(200)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        icpTransformFilter = vtkTransformPolyDataFilter()
        icpTransformFilter.SetInputData(source)

        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()

        # source mesh transformed
        transformedSource = icpTransformFilter.GetOutput()

        # write transformed vertices to file
        point = [0, 0, 0]
        v_trans = v.copy()

        for i in range(len(v)):
            transformedSource.GetPoint(i, point)
            v_trans[i, :] = point

        file_out = os.path.join(data_out, stl_files[j].split('/')[-1])
        writeTriangularSurfaceMesh(v_trans, f, file_out)



def iterative_closest_point_with_node_correspondence(data_base, data_out):
    '''
    Alignment via iterative closest point (IPC) algorithm using vtk modules
    (https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/)
    Algorithm following principal of rigid transformation
    Only works for grids with node correspondence, calculate mean shape and use as target file

    Parameters
    ----------
    data_base : string
        absolute path of directory of data to align.
    data_out : string
        absolute path of directory to save aligned files.

    Returns
    -------
    None.
    '''

    stl_files = glob.glob(data_base + "*.stl")
    stl_files = sorted(stl_files)
    # print(stl_files)

    # initialize vertices matrix for finding mean shape as target mesh
    v_init, f_init = readTriangularSurfaceMesh(stl_files[0])
    vertices = np.empty((len(v_init[:, 0]) * len(v_init[0, :]), len(stl_files)))

    k = 0
    for stl_file in stl_files:
        v, f = readTriangularSurfaceMesh(stl_file)
        v_1d = np.array(np.reshape(v, (len(v[0,:])*len(v[:,0]), 1)))
        vertices[:, k] = v_1d[:, 0]
        k += 1

    # compute mean shape
    v_mean = vertices.mean(axis=1)

    targetPoints = vtkPoints()
    targetVertices = vtkCellArray()

    for i in range(len(v_mean)):
        tp_id = targetPoints.InsertNextPoint(v_mean[i, :])
        targetVertices.InsertNextCell(1)
        targetVertices.InsertCellPoint(tp_id)

    target = vtkPolyData()
    target.SetPoints(targetPoints)
    target.SetVerts(targetVertices)

    print("\nnumber of transformed files ", len(stl_files))
    # print("\ntarget file ", stl_files[idx_target], "\n")

    for j in range(len(stl_files)):
        print("run IPC algorithm for file nr ", j)

        v, f = readTriangularSurfaceMesh(stl_files[j])

        sourcePoints = vtkPoints()
        sourceVertices = vtkCellArray()

        # read source file
        for i in range(len(v)):
            sp_id = sourcePoints.InsertNextPoint(v[i, :])
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(sp_id)

        source = vtkPolyData()
        source.SetPoints(sourcePoints)
        source.SetVerts(sourceVertices)

        # ============ run ICP ==============
        icp = vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations(20)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        icpTransformFilter = vtkTransformPolyDataFilter()
        icpTransformFilter.SetInputData(source)

        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()

        # source mesh transformed
        transformedSource = icpTransformFilter.GetOutput()

        # write transformed vertices to file
        point = [0, 0, 0]
        v_trans = v.copy()

        for i in range(len(v)):
            transformedSource.GetPoint(i, point)
            v_trans[i, :] = point

        file_out = os.path.join(data_out, stl_files[j].split('/')[-1])
        writeTriangularSurfaceMesh(v_trans, f, file_out)

    return


def adaptE1E2CoordinateSystem(base_org,e1_orientation):
    base = base_org.copy()
    e1 = e1_orientation - np.dot(e1_orientation, base[-1])*base[-1]
    base[0] = e1/np.linalg.norm(e1)
    e2 = np.cross(base[2],base[0])
    base[1] = e2/np.linalg.norm(e2)

    return base


def createCoordinateSystem(pts):
    '''
    Creates Coordinate system for point cloud via linear PCA

    Parameters
    ----------
    pts : (N, dim) numpy nd array
        coordinates of data points.

    Returns
    -------
    base: list of (dim,) numpy nd arrays
        base vectors of coordinate system calculated.
    ctr: (dim,) numpy array
        origin of coordinate system.
    '''
    # shape (3,-1) ?
    # pts = np.reshape(pts, (-1, np.shape(pts)[0]))

    # calculate center
    ctr = pts.mean(axis=0)

    # calculate point vectors from center
    vector = pts - ctr #[np.newaxis,:]
    # compute plane normal
    E = np.linalg.svd(np.dot(vector.T, vector))[0]
    e1 = E[:,0]
    e2 = E[:,1]
    e3 = E[:,2]

    return [e1,e2,e3],ctr


def getFaceCenter(vertices,faces):
    '''
    compute the center of all faces

    Parameters
    ----------
    vertices : (N, dim) numpy nd array
        coordinates of data points of shape.
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.

    Returns
    -------
    ctr_faces : (M, dim) numpy nd array
        coordinates of face centers.
    '''
    
    ctr_faces = np.empty((len(faces),3))
    for i in range(len(faces)):
        i1,i2,i3 = faces[i,0], faces[i,1], faces[i,2]
        v1,v2,v3 = vertices[i1,:], vertices[i2,:], vertices[i3,:]
        v1,v2,v3 = np.array(v1),np.array(v2),np.array(v3)
        tri = np.stack((v1,v2,v3), axis=0)
        ctr = np.mean(tri, axis=0)
        ctr_faces[i,:] = ctr
    
    return ctr_faces

    

def getOrderedBoundaryCurve(vertices,faces,*nBoundaries):
    '''
    Returns coordinates of boundary vertices of mesh. Works if only one enclosed
    boundary exists. Only for triangulation

    Parameters
    ----------
    vertices : (n,3) numpy ndarray shape
        numpy array of vertices.
    faces : (m,3) numpy ndarray shape
        numpy array of faces.

    Returns
    -------
    vertices_boundary: (k,3) numpy ndarray shape
        coordinates of boundary vertices.
   '''
    # if no info about nBoundaries provided, assumed that solely one exists
    if nBoundaries:
        nBoundaries = nBoundaries[0]
    else:
        nBoundaries = 1
        
    
    # get and order edges of mesh
    edges = np.concatenate([faces,faces],axis=1).reshape(faces.size,2)
    edges = np.sort(edges)
    
    # compute boundary points
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    edges_boundary = unique[counts==1,:]
    
    vertices_boundary_all = []
    faces_id_all = []
    for i in range(nBoundaries):
        # get ordered boundary points
        index_start = edges_boundary[0,0]
        faces_id = []
        faces_id.append(edges_boundary[0,0])
        faces_id.append(edges_boundary[0,1])
        vertices_boundary = []
        vertices_boundary.append(vertices[edges_boundary[0,0],:])
        vertices_boundary.append(vertices[edges_boundary[0,1],:])
        index = edges_boundary[0,1]
        edges_boundary = np.delete(edges_boundary, (0), axis=0)
        
        # loop until all covered
        while index != index_start:
            location = np.where(edges_boundary==index)
            row = location[0][0]
            column = location[1][0]
            
            index = edges_boundary[row,column-1]
            if index==index_start:
                edges_boundary = np.delete(edges_boundary, (row), axis=0)
                continue
            faces_id.append(index)
            vertices_boundary.append(vertices[index,:])
            edges_boundary = np.delete(edges_boundary, (row), axis=0)
            
        vertices_boundary_all.append(vertices_boundary)
        faces_id_all.append(faces_id)
        
    if nBoundaries==1:
        return np.array(vertices_boundary_all[0]).T #, np.array(faces_id_all[0]).T
    else:
        return vertices_boundary_all, faces_id_all
    


def calculateDeformation(vertices0, vertices1, max_dim):
    '''
    Calculates the distance of the vertices between to corresponding shapes

    Parameters
    ----------
    vertices0 : (N, dim) numpy nd array
        coordinates of data points of shape 0.
    vertices1 : (N, dim) numpy nd array
        coordinates of data points of shape 1.
    max_dim : float, integer
        maximum reference length for computing relative deformation.

    Returns
    -------
    dist_rel : (N, ) numpy array
        relative distance between vertices of shape 0 and 1.
    '''
    # calculate deformation between msh0 and msh1
    dev = vertices0 - vertices1

    dist = np.empty([len(dev[:,0])])
    for i in range(len(dev[:,0])):
        dist[i] = np.linalg.norm(dev[i,:],2)

    dist_rel = dist/max_dim*100

    return dist_rel