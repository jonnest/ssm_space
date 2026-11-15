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

The routine reads a .txt file, which is edited by the main routine, the print
output is read in from the python interpreter.

'''

import os
import pymeshlab



def checkMeshQuality(filename, limit_self):
    """
    Check mesh quality  Meshlab.
    
    Parameters
    ----------
    filename : string
        name of file. 
        
    Returns
    -------
    good_quality : boolean
        Is mesh of good quality?
    no_intersect : boolean
        Is mesh intersect free?
    
    """   
    
    # check topological measures
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.get_topological_measures()
    if (mesh["boundary_edges"] != 160) or (mesh["connected_components_number"] != 1) or \
        (mesh["edges_number"] != 4640) or (mesh["faces_number"] != 3040) or (mesh["genus"] != 0) or \
        (mesh["incident_faces_on_non_two_manifold_edges"] != 0) or \
        (mesh["incident_faces_on_non_two_manifold_vertices"] != 0) or (mesh["is_mesh_two_manifold"] is not True) or \
        (mesh["non_two_manifold_edges"] != 0) or (mesh["non_two_manifold_vertices"] != 0) or \
        (mesh["number_holes"] != 2) or (mesh["unreferenced_vertices"] != 0) or (mesh["vertices_number"] != 1600):
        good_quality = False
    else:
        good_quality = True
        
    # check self intersections
    ms.compute_selection_by_self_intersections_per_face()
    ms.apply_selection_inverse()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()
    mesh = ms.get_topological_measures()
    if (mesh["vertices_number"] > limit_self) and (mesh["vertices_number"] != 1600):
    ##if (mesh["vertices_number"] != 0) and (mesh["vertices_number"] != 1600):
        no_intersect = False
    else:
        no_intersect = True
        
    return good_quality, no_intersect


###############################################################################
# read path from txt file 
# workaround required due to incompatibility of PyMeshLab in Spyder
# this_mesh = open("pymeshlab/pymeshlab_file.txt", 'r').read()
file = open("pymeshlab/pymeshlab_file.txt", 'r')
lines = file.readlines()
 
this_mesh = lines[0][:-1]
limit_self = int( lines[1] ) 
    
# check mesh quality and for self-intersections
good_quality, no_intersect = checkMeshQuality(this_mesh, limit_self )

# generate dictionary to print
results = {
    "no_intersect" : no_intersect,
    "good_quality" : good_quality
}

###############################################################################
# print results as dictionary expressed as a string
print(results)

