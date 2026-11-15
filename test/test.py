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
from ssm.ssm_mv_tomtec.infile import * 

from tools.io_utils import *

def flexible_function(*args):
    # Define the default variable names
    var_names = ["arg1", "arg2", "arg3", "arg4", "arg5"]

    # Assign the optional arguments to the variable names
    var = dict(zip(var_names, args))

    # Do something with the optional arguments
    print("Optional arguments:")
    for var_name, value in var.items():
        print(f"{var_name}: {value}")

    return var 

# Example usage of the function
# var = flexible_function(10, 20, 30, 10, 20, 30)

def find_row_with_values(arr, value1, value2):
    # Create a boolean mask for rows containing value1
    mask_value1 = (arr == value1)

    # Create a boolean mask for rows containing value2
    mask_value2 = (arr == value2)

    # Combine the two masks with a logical AND operation to find rows with both values
    rows_with_values = np.logical_or(mask_value1, mask_value2)
    
    # Use np.where() to get the indices of rows with the desired values
    indices = np.where(np.any(rows_with_values, axis=1))[0]

    return indices


def detect_anterior_artefact(mesh, start, width, rows, nr_v_circ, ang_threshold):
    '''
    Detect leaflet artefact. Define search area
    
    Id's of search area:
        
    end-rows*nr_v_circ x x x x x x x x x x x x start-rows*nr_v_circ
           x           x x x x x x x x x x x x          x
           x           x x x x x x x x x x x x          x
           x           x x x x x x x x x x x x          x
           x           x x x x x x x x x x x x          x
          end          x x x x x x x x x x x x        start
     
    Parameters
    ----------
    mesh : string
        Absolute path of mesh file to analyse.
    start_id : integer
        Vertex id of start of the search box, see search area definition.
    width : integer
        Width of the search box, see search area definition.
    rows : integer
        Number of rows of the search box, see search area definition.
    ang_threshold : float
        Threshold for the angle between the normal vectors of two neighboring faces. 
    
    Returns
    -------
    artefact_bool : bool
        Does the mesh has an artefact on the anterior leaflet ?
    max_angle : float
        Maximum angle between two normal vectors of two neighboring faces.
        
    '''
    # read mesh
    v, f = readTriangularSurfaceMesh(mesh)
    
    # get all vertex ids in search area
    # search faces containing two of the 
    ids = []
    angles = []
    for i in range(rows):
        for j in range(width-2):
            # vertex id of starting point to define the triangles tri1 and tri2
            ids = start - i*nr_v_circ + j 
            
            # index of triangles in face matrix
            idx_tri1 = find_row_with_values(f, ids  , ids+1)[0]
            idx_tri2 = find_row_with_values(f, ids+1, ids+2)[0]
            
            # vertex indices defining the triangles tri1 and tri2
            v11, v12, v13 = f[idx_tri1]
            v21, v22, v23 = f[idx_tri2]
            
            # compute normal vector of the triangles
            n1 = np.cross( v[v13]-v[v11], v[v12]-v[v11] )
            n2 = np.cross( v[v23]-v[v21], v[v22]-v[v21] )
            
            # Calculate the angle in radians using the dot product and magnitudes
            ang_rad = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) *  np.linalg.norm(n2)))
            ang_deg = np.degrees(ang_rad)
            
            if abs(ang_deg) > 90 : ang_deg = ang_deg - 90 
            
            # if abs(ang_deg) > 20 : print(ang_deg)
            
            angles.append(ang_deg)
            
    # maximum angle
    max_angle = max(angles)
    
    artefact_bool = False
    # if the maximum angle is larger than the threshold, then an artefact on antorior leaflet exists
    if max_angle > ang_threshold : artefact_bool = True
    
    return artefact_bool, max_angle


project_path = os.path.abspath(workpath + "/" + workspace + "/")
data_path = os.path.abspath(project_path +  "/" + "OBJ" + "/")


# mesh = os.path.abspath(data_path + "/1209/middia/MV_middia_aligned.obj")

# mesh = os.path.abspath(workpath + "/distorted_grids/shape_00055.obj")
# artefact_bool, max_angle = detect_anterior_artefact(mesh, 1550, 20, 10, 80, 30 )

test_path = os.path.abspath(workpath + "/distorted_grids")
print(test_path)

for dirpath, subdirs, files in os.walk(test_path):
    for file in files:
        mv_mesh = os.path.join(dirpath, file)
        artefact_bool, max_angle = detect_anterior_artefact(mv_mesh, 1550, 20, 10, 80, 30 )
                    
        print(file, "      ",max_angle)












