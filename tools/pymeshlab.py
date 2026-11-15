################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import pymeshlab


def checkMeshQuality(filename):
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
    mesh = ms.compute_topological_measures()
    if (mesh["boundary_edges"] != 160) or (mesh["connected_components_number"] != 1) or (mesh["edges_number"] != 4640) or (mesh["faces_number"] != 3040) or (mesh["genus"] != 0) or (mesh["incident_faces_on_non_two_manifold_edges"] != 0) or (mesh["incident_faces_on_non_two_manifold_vertices"] != 0) or (mesh["is_mesh_two_manifold"] is not True) or (mesh["non_two_manifold_edges"] != 0) or (mesh["non_two_manifold_vertices"] != 0) or (mesh["number_holes"] != 2) or (mesh["unreferenced_vertices"] != 0) or (mesh["vertices_number"] != 1600):
        good_quality = False
    else:
        good_quality = True
        
    # check self intersections
    ms.select_self_intersecting_faces()
    ms.invert_selection()
    ms.delete_selected_faces()
    ms.remove_unreferenced_vertices()
    mesh = ms.compute_topological_measures()
    if (mesh["vertices_number"] != 0) and (mesh["vertices_number"] != 1600):
        no_intersect = False
    else:
        no_intersect = True
        
    return good_quality,no_intersect


def close_all_holes(inputName,outputName):
    '''
    Close all holes of given mesh

    Parameters
    ----------
    inputName : string
        input mesh file with holes
    outputName : string
        output mesh file with all holes closed

    Returns
    -------

    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    ms.meshing_close_holes(maxholesize = 50000)
    ms.save_current_mesh(outputName)


def geometric_measure(inputName):
    '''
    Get geometric measurements of input mesh (.stl)

    Parameters
    ----------
    inputName : string
        input mesh file (.stl) to measure

    Returns
    -------
    gm : dictionary
        geometric measures of input mesh
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    gm = ms.get_geometric_measures()
    return gm


def measure_mesh(inputName):
    '''
    Get bounding box measures of a mesh

    Parameters
    ----------
    inputName : string
        path of mesh to analyze
    Returns
    -------
    bbox : dictionary
        bounding box measures
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    measures = ms.get_geometric_measures()
    bbox = measures['bbox']
    return bbox


def isotropic_remeshing(inputName,outputName,targetlen):
    '''
    Remeshing input .stl mesh file using target length

    Parameters
    ----------
    inputName : string
        path to mesh file to remesh
    outputName : string
        path of output mesh file of remeshed grid
    targetlen : float
        percentage of characteristic length of mesh, recommend isotropic_remeshing_target()

    Returns
    -------

    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    ms.remeshing_isotropic_explicit_remeshing(targetlen=targetlen)
    ms.save_current_mesh(outputName)


def isotropic_remeshing_target(file_name,target,string_ext_filename='',forced=False):
    '''
    Remesh given .stl file, requires target number of desired node number

    Parameters
    ----------
    file_name : string
        mesh file to remesh
    target : integer
        desired number of nodes for remeshed file
    string_ext_filename :
        string extension attached to filename (optional, default is no additional string '')
    forced : if remeshing file which already fulfills target number of nodes

    Returns
    -------
    output_name : string
        new name of remeshed esh file

    '''
    # loop for remeshing and hitting target
    delta = 50
    # lower boundary
    v_min = target - delta
    # upper boundary
    v_max = target + delta
    output_name = string_ext_filename + file_name
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_name)
    mesh = ms.current_mesh()
    nr_v = mesh.vertex_number()
    ms.save_current_mesh(output_name)

    print('number of vertices ', nr_v)
    print('target number in range ', target-delta, ' - ', target+delta)
    # initial edge percentage and optimization step
    per = 2.0
    opt = 0.1
    # while number of vertices is not in target range - do remeshing and find fitting parameters
    while nr_v < v_min or nr_v > v_max or forced == True:
        if nr_v < v_min:
            per = per - opt
        elif nr_v > v_max:
            per = per + opt

        print('remeshing with per = ', per)
        # remesh using different function
        isotropic_remeshing(output_name, output_name, targetlen=pymeshlab.Percentage(per))
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(output_name)
        mesh = ms.current_mesh()
        nr_v = mesh.vertex_number()
        print('new number of vertices ', nr_v, '\n')

        # when hitting the target turn off forced remeshing
        if nr_v > v_min and nr_v < v_max:
            forced = False
        # when edge percentage is negative - optimization did not work - decrease step size "opt"
        if per < 0:
            print('did not find solution')
            break

    return output_name


def compute_hausdorff(mesh1, mesh2):
    '''
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


def face_quality(inputName):
    '''
    Get face quality of given mesh (.stl file)

    Parameters
    ----------
    inputName : string
        mesh file (.stl file) to analyze mesh quality

    Returns
    -------
    ratio : dictionary
        mean ratio of all faces; r = Area/(a*a + b*b + c*c) with a,b,c edge lengths
    hist : dictionary
        histogram directory of mean ratio
    stats : dictionary
        directory of statistical overview of mean ratio of all faces
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    ratio = ms.compute_scalar_by_aspect_ratio_per_face(metric='Mean ratio')
    # stats = ms.get_scalar_histogram_per_face()
    hist = ms.get_scalar_histogram_per_face()
    stat = ms.get_scalar_statistics_per_face()
    return ratio, hist, stat


def face_area(inputName):
    '''
    Get face area of given mesh (.stl file)

    Parameters
    ----------
    inputName : string
        mesh file (.stl file) to analyze mesh quality

    Returns
    -------
    stat : dictonary
        scalar statistics per face
    '''
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(inputName)
    tri = ms.compute_scalar_by_aspect_ratio_per_face(metric='Area')
    # stats = ms.get_scalar_histogram_per_face()
    # hist = ms.get_scalar_histogram_per_face()
    stat = ms.get_scalar_statistics_per_face()
    return stat
