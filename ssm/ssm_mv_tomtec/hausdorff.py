################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
This file is executed in the Windows Shell.
The workaround is necessary for a consistent use of the package PyMeshLab, due 
to its incompatibility with Spyder.


'''

import os
import pymeshlab
import pyvista as pv


def readTriangularSurfaceMesh(path):
    '''
    read triangluar mesh from path

    Parameters
    ----------
    path : string
        path and name of mesh file.

    Returns
    -------
    vertices : (N, dim) numpy nd array
        coordinates of data points of shape.
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.
    '''
    mesh = pv.read(path)
    vertices,faces = mesh.points,(mesh.faces.reshape(int(mesh.faces.shape[0]/4),4)[:,1:])

    return vertices,faces


def compute_hausdorff(mesh1, mesh2):
    '''
    Compute the Hausdorff distance between the vertices of two meshes.
    
    PyMeshlab Filter: "Compute the Hausdorff Distance between two layers, sampling one of the two 
    and finding for each sample the closest point over the other mesh. 
    Distance is stored in vertex quality of the sampled mesh."
    
    Parameters
    ----------
    mesh1 : string
        path to mesh file 1
    mesh2 : string
        path mesh file 2 to compare with

    Returns
    -------
    hd_max : float
        maximum hausdorff distance
    hd_mean : float
        average hausdorff distance
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh1)
    ms.load_new_mesh(mesh2)
    
    # get hausdorff distance returns dictornary : 'RMS', 'diag_mesh_0', .. , 'max', 'mean', 'min', 'n_samples'
    hd_dic = ms.get_hausdorff_distance()
    hd_max = hd_dic['max']
    hd_mean = hd_dic['mean']
    
    return hd_max, hd_mean


###############################################################################
# read path from txt file 
# workaround required due to incompatibility of PyMeshLab in Spyder
# this_mesh = open("pymeshlab/pymeshlab_file.txt", 'r').read()

mesh1 = "pymeshlab/mesh1.stl"
mesh2 = "pymeshlab/mesh2.stl"


# check mesh quality and for self-intersections
hd_max, hd_mean = compute_hausdorff(mesh1, mesh2)

# generate dictionary to print
results = {
    "hd_max" : hd_max,
    "hd_mean" : hd_mean
}

###############################################################################
# print results as dictionary expressed as a string
print(results)
