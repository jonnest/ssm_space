################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Filter cohort of synthetic geometries.

- use pymeshlab to select self-intersecting meshes
  workaround: create file for executing pymeshlab, does not work with spyder
              or use PyCharm 
              
- filter data using thresholds of metrics

    
'''
import os 
import subprocess
import shutil
import ast
import sys
import glob
import numpy as np
import scipy as scp

from tools.pointclouds import getOrderedBoundaryCurve
from tools.io_utils import readTriangularSurfaceMesh, writeTriangularSurfaceMesh, \
                            writeOBJ


def remove_filtered_synthetic_data(syn_path, name, filter_type, *args):
    '''
    Filter synthetic meshes and move to filtered folder.

    Parameters
    ----------
    syn_path : string
        Path to synthetic data folder, walking through folder.
    name : string
        Name of the data, e.g. the state (diastolic or systolic).
        
    Returns
    -------
    None.

    '''
    
    print("\nFind self-intersecting meshes in synthetic data...")
    
    if filter_type == "intersect":
        # variable names
        var_names = ["limit_self"]
        # Assign the optional arguments to the variable names
        var = dict(zip(var_names, args))
            
    elif filter_type == "artefact":
        # variable names
        var_names = ["start", "width", "rows", "nr_v_circ", "ang_threshold"]
        # Assign the optional arguments to the variable names
        var = dict(zip(var_names, args))
    
    # counters for dia and sys files
    nr_shapes = 0
    
    for dirpath, subdirs, files in os.walk(syn_path):
        for file in files:
            if "shape" in file and "anterior" not in file and \
                "posterior" not in file and "orifice" not in file and \
                "intersect" not in dirpath and "filter" not in dirpath:
                # mesh file of loop
                mesh = os.path.join(dirpath, file)
                
                # new folder
                filter_path = dirpath + f"_{filter_type}"
    
                if not os.path.exists(filter_path):
                    os.makedirs(filter_path)

                # loop over values in dictonary
                if f"{name}" in dirpath:
                    
                    # filter intersections
                    if filter_type == "intersect":
                        # get result of intersection bool
                        result_dict = detect_self_intersections(mesh, var["limit_self"])
                        # if intersection in mesh, then move mesh
                        if result_dict["no_intersect"] == False:
                            shutil.move(mesh, filter_path)
                            nr_shapes += 1

                    if filter_type == "artefact":
                        v, f = readTriangularSurfaceMesh(mesh)
                        # get result of intersection bool
                        artefact_bool, max_angle = detect_anterior_artefact(v, f, 
                                                                            var["start"],
                                                                            var["width"],
                                                                            var["rows"],
                                                                            var["nr_v_circ"],
                                                                            var["ang_threshold"])
                        
                        # if artefact in mesh, then move mesh
                        if artefact_bool == True:
                            shutil.move(mesh, filter_path)
                            nr_shapes += 1

                    
    print(f"\nNumber of self-intersecting synthetic meshes {name} :   ", nr_shapes)

    return
    
    
def detect_anterior_artefact(v, f, start, width, rows, nr_v_circ, ang_threshold):
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
    v : (N, dim) numpy nd array
        coordinates of data points of shape.
    f : (M, dim) numpy nd array
        definition of triangles via vertex indices.
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
            
            # use always the small angle between the normal vectors, never larger than 90 degs
            if abs(ang_deg) > 90 : ang_deg = ang_deg - 90 
            
            angles.append(ang_deg)
            
    # maximum angle
    max_angle = max(angles)
    
    artefact_bool = False
    # if the maximum angle is larger than the threshold, then an artefact on antorior leaflet exists
    if max_angle > ang_threshold : artefact_bool = True
        
    return artefact_bool, max_angle



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



def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two 3D points
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))


def dynamic_time_warping(curve1, curve2):
    # Initialize the cost matrix with large values
    dtw = np.full((len(curve1), len(curve2)), float('inf'))

    dtw[0,0] = 0
    
    for i in range(1, len(curve1)):
        for j in range(1, len(curve2)):
            cost = euclidean_distance(curve1[i], curve2[j])
            
            dtw[i, j] = cost + min(dtw[i-1, j  ],   # insertion
                                   dtw[i  , j-1],   # deletion
                                   dtw[i-1, j-1])   # match

    # Return the DTW distance (the value at the bottom-right cell of the accumulated cost matrix)
    return dtw[-1, -1]


def get_dynamic_time_warpings(dia_shapes, sys_shapes, init_mesh_files, metric_path, name): 
    '''
    Execute dynamic time warping for annulus boundary curves between systolic
    and diastolic states.
    
    Procedure: 
        - Extract the annuli of the states
        - Move to the global center (origin [0,0,0])
        - Triangulate surface using the center; necessary to import for Pymeshlab
        - Save both geometries in temporary folder in the Code directory

    Parameters
    ----------
    dia_shapes : (N, M) numpy nd array
        Shape matrix of diastolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    sys_shapes : (N, M) numpy nd array
        Shape matrix of systolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    init_mesh_files : list of strings
        Path to mesh file for initialization of the shape matrices and faces.
    metric_path : string
        Absolute path to metrices directory.
    name : string
        Name of the data type, e.g. "org" or "syn".

    Returns
    -------
    dtws : (M,) numpy array
        Array of DTWs for all pairs of states M.

    '''
    
    print(f"\nCompute DTWs of annuli for {name} data...\n")    

    if len(sys_shapes) != len(dia_shapes) :
        print("\n!!! Warning !!!\n\nSynthetic shape matrices of diastolic and systolic need to be of same size !\n")
        print("*** End of program ***\n")
        sys.exit()
    
    # initialization of faces and vertices
    # here : in case faces f are equal for diastolic and systolic state
    v, f = readTriangularSurfaceMesh(init_mesh_files[0])   

    # initialize hd array
    dtws = np.empty(len(dia_shapes[0,:]))
    
    # loop over all shape pairs
    for i in range(len(dia_shapes[0,:])):
        # get 1d vertices
        v_1d_dia = dia_shapes[:,i]
        v_1d_sys = sys_shapes[:,i]
        
        # reshape vertices to 2d array
        v_2d_dia = np.array(np.reshape(v_1d_dia, ( int(len(v_1d_dia)/3), -1)) )
        v_2d_sys = np.array(np.reshape(v_1d_sys, ( int(len(v_1d_sys)/3), -1)) )
        
        # get boundary, first bnd is annulus
        bnd_dia, f_dia = getOrderedBoundaryCurve(v_2d_dia, f, 2)
        bnd_sys, f_sys = getOrderedBoundaryCurve(v_2d_sys, f, 2)
        
        # convert to numpy array
        annu_dia = np.array(bnd_dia[0])
        annu_sys = np.array(bnd_sys[0])

        
        # move to center, calculate point vectors from center
        annu_dia = annu_dia - annu_dia.mean(axis=0)
        annu_sys = annu_sys - annu_sys.mean(axis=0)
        
        dtws[i] = dynamic_time_warping(annu_dia, annu_sys)

    # write all means in txt file
    all_means_file = os.path.join(metric_path, f"{name}_dtws.txt")
    np.savetxt(all_means_file, dtws)
    
    return dtws



def get_hausdorff_means(dia_shapes, sys_shapes, init_mesh_files, metric_path, name): 
    '''
    Get the Hausdorff distance measures for annulus boundary curves between systolic
    and diastolic states.
    
    Procedure: 
        - Extract the annuli of the states
        - Move to the global center (origin [0,0,0])
        - Triangulate surface using the center; necessary to import for Pymeshlab
        - Save both geometries in temporary folder in the Code directory

    Parameters
    ----------
    dia_shapes : (N, M) numpy nd array
        Shape matrix of diastolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    sys_shapes : (N, M) numpy nd array
        Shape matrix of systolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    init_mesh_files : list of strings
        Path to mesh file for initialization of the shape matrices and faces.
    metric_path : string
        Absolute path to metrices directory.
    name : string
        Name of the data type, e.g. "org" or "syn".

    Returns
    -------
    hd_means : (M,) numpy array
        Array of Hausdorff distance means for all pairs of states M.

    '''
    
    print(f"\nCompute Hausdorff means of annuli for {name} data...\n")    

    if len(sys_shapes) != len(dia_shapes) :
        print("\n!!! Warning !!!\n\nSynthetic shape matrices of diastolic and systolic need to be of same size !\n")
        print("*** End of program ***\n")
        sys.exit()
    
    # initialization of faces and vertices
    # here : in case faces f are equal for diastolic and systolic state
    v, f = readTriangularSurfaceMesh(init_mesh_files[0])
    
    # write path to file in specific path folder
    pymeshlab_folder = "pymeshlab"
    pymeshlab_file_1 = "mesh1.stl"
    pymeshlab_file_2 = "mesh2.stl"
    
    # initialize hd array
    hd_means = np.empty(len(dia_shapes[0,:]))
    
    # loop over all shape pairs
    for i in range(len(dia_shapes[0,:])):
        # get 1d vertices
        v_1d_dia = dia_shapes[:,i]
        v_1d_sys = sys_shapes[:,i]
        
        # reshape vertices to 2d array
        v_2d_dia = np.array(np.reshape(v_1d_dia, ( int(len(v_1d_dia)/3), -1)) )
        v_2d_sys = np.array(np.reshape(v_1d_sys, ( int(len(v_1d_sys)/3), -1)) )
        
        # get boundary, first bnd is annulus
        bnd_dia, f_dia = getOrderedBoundaryCurve(v_2d_dia, f, 2)
        bnd_sys, f_sys = getOrderedBoundaryCurve(v_2d_sys, f, 2)
        
        # convert to numpy array
        annu_dia = np.array(bnd_dia[0])
        annu_sys = np.array(bnd_sys[0])

        f_dia = np.array(f_dia[0])
        f_sys = np.array(f_sys[0])
        
        # move to center, calculate point vectors from center
        annu_dia = annu_dia - annu_dia.mean(axis=0)
        annu_sys = annu_sys - annu_sys.mean(axis=0)
        
        # add center to last row for triangulation
        annu_dia = np.vstack((annu_dia, annu_dia.mean(axis=0)))
        annu_sys = np.vstack((annu_sys, annu_sys.mean(axis=0)))
        
        # initialization
        f_annu = np.empty((len(f_dia),3))
        
        # loop over faces
        for j in range(len(f_dia)-1):
            f_annu[j,:] = [j, j+1, len(f_dia)]
        
        # last row to close the annulus
        f_annu[-1,:] = [f_dia[-1], f_dia[0], len(f_dia)]
        
        # convert to integers
        f_annu = np.array(f_annu, dtype=int)
        
        # paths to save temporary mesh
        pathfile_pymeshlab_1 = os.path.abspath(pymeshlab_folder + "/" + pymeshlab_file_1)
        pathfile_pymeshlab_2 = os.path.abspath(pymeshlab_folder + "/" + pymeshlab_file_2)
        
        writeTriangularSurfaceMesh(annu_dia, f_annu, pathfile_pymeshlab_1)
        writeTriangularSurfaceMesh(annu_sys, f_annu, pathfile_pymeshlab_2)   
    
        # execute pymeshlab file to find self intersections
        output = subprocess.getoutput("python hausdorff.py")
        
        # save shell output in dictionary
        result_dict = ast.literal_eval(output)
        
        # print("\n\nAfter the subprocess :  ", result_dict) 
        
        hd_means[i] = result_dict["hd_mean"]

    # write all means in txt file
    all_means_file = os.path.join(metric_path, f"{name}_hausdorff_distance.txt")
    np.savetxt(all_means_file, hd_means)
    
    return hd_means


def get_procrustes_disparity(dia_shapes, sys_shapes, init_mesh_files, metric_path, name): 
    '''
    Get the Procrustes disparity between the annulus boundary curves between systolic
    and diastolic states. The measure is computed via a Procrustes alignemnt, 
    using scaling and rotation.
    The disparity ( M^2 = sum(data1-data2)^2 ) here is a measure for the similarity
    of the annuli.
    - M^2 = 0 : data sets are similiar
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html

    Parameters
    ----------
    dia_shapes : (N, M) numpy nd array
        Shape matrix of diastolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    sys_shapes : (N, M) numpy nd array
        Shape matrix of systolic state with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    init_mesh_files : list of strings
        Path to mesh file for initialization of the shape matrices and faces.
    metric_path : string
        Absolute path to metrices directory.
    name : string
        Name of the data type, e.g. "org" or "syn".

    Returns
    -------
    d_all : (M,) numpy array
        Array of disparity for all pairs of states M.

    '''
    
    print(f"\nCompute the Procrustes disparity of annuli for {name} data...\n")    

    if len(sys_shapes) != len(dia_shapes) :
        print("\n!!! Warning !!!\n\nSynthetic shape matrices of diastolic and systolic need to be of same size !\n")
        print("*** End of program ***\n")
        sys.exit()
    
    # initialization of faces and vertices
    # here : in case faces f are equal for diastolic and systolic state
    v, f = readTriangularSurfaceMesh(init_mesh_files[0])
    
    # initialize hd array
    d_all = np.empty(len(dia_shapes[0,:]))
    
    # loop over all shape pairs
    for i in range(len(dia_shapes[0,:])):
        # get 1d vertices
        v_1d_dia = dia_shapes[:,i]
        v_1d_sys = sys_shapes[:,i]
        
        # reshape vertices to 2d array
        v_2d_dia = np.array(np.reshape(v_1d_dia, ( int(len(v_1d_dia)/3), -1)) )
        v_2d_sys = np.array(np.reshape(v_1d_sys, ( int(len(v_1d_sys)/3), -1)) )
        
        # get boundary, first bnd is annulus
        bnd_dia, f_dia = getOrderedBoundaryCurve(v_2d_dia, f, 2)
        bnd_sys, f_sys = getOrderedBoundaryCurve(v_2d_sys, f, 2)
        
        # convert to numpy array
        annu_dia = np.array(bnd_dia[0])
        annu_sys = np.array(bnd_sys[0])

        f_dia = np.array(f_dia[0])
        f_sys = np.array(f_sys[0])
        
        # move to center, calculate point vectors from center
        annu_dia = annu_dia - annu_dia.mean(axis=0)
        annu_sys = annu_sys - annu_sys.mean(axis=0)
        
        # compute disparity
        _, _, disparity_i = scp.spatial.procrustes(annu_dia, annu_sys)     
        d_all[i] = disparity_i

    # write all means in txt file
    all_disparities_file = os.path.join(metric_path, f"{name}_procrustes_disparity.txt")
    np.savetxt(all_disparities_file, d_all)
    
    return d_all


def find_outliers(data_org, data_syn, metrices_to_filter, iqr_stretch=0):
    '''
    Find synthetic outliers with respect to the original cohort data.
    Outliers are detected, when outside of a certain range (reference is the interquartile range = IQR):
    Lower bound: q1 - (1.5 + iqr_stretch) * IQR
    Upper bound: q3 + (1.5 + iqr_stretch) * IQR

    Parameters
    ----------
    data_org : (N, M) numpy nd array
        Parameter distribution sets of original data.
        N: number of shapes.
        M: number of parameters, e.g. metrics.
    data_syn : (N, M) numpy nd array
        Parameter distribution sets of synthetic data.
        N: number of shapes.
        M: number of parameters, e.g. metrics.
    metrices_to_filter : dictionary
        Contains all computed metrics (given in the infile), for every computed metric the item indicates if filtering
        applies on this parameter (no==0, yes==1).
    iqr_stretch : float
        Factor to increase the filtering range: (1.5 + iqr_stretch) * iqr.

    Returns
    -------
    outliers_idx : (O,) numpy array
        Array with indices of the synthetic cohort to filter.

    '''    
    # loop over the metrics
    outliers_idx = np.empty(0)
    
    # loop over metrics which are considered for filtering
    for i, item in enumerate(metrices_to_filter.items()):
        # if filter flag is set to 1, filtering considered
        if item[1] == 1:
            # assign your quartiles, limits and iq3 of the real data
            q1, q3 = np.percentile(data_org[:,i], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 + iqr_stretch) * iqr
            upper_bound = q3 + (1.5 + iqr_stretch) * iqr
            
            # create conditions to isolate the outliers
            temp_idx = np.where((data_syn[:,i] < lower_bound) | (data_syn[:,i] > upper_bound))
            
            outliers_idx = np.append(outliers_idx, temp_idx)
        

    # remove duplicates
    outliers_idx = [*set(outliers_idx)]
    outliers_idx = sorted(np.array(outliers_idx, dtype=int))
    
    return outliers_idx



def filter_metrics(syn_path, data_org, data_syn, iqr_stretch, metrices_to_filter, 
                   name):
    '''
    Filtering the synthetic shapes using the interquartile range with (Q1/Q3) to detect outliers.
    All outliers (measured according the original data set) are moved into a separate filter folder.

    Parameters
    ----------
    syn_path : string
        Absolute path to the directory of the synthetic data.
    data_org : (N, M) numpy nd array
        Parameter distribution sets of original data.
        N: number of shapes.
        M: number of parameters, e.g. metrics.
    data_syn : (N, M) numpy nd array
        Parameter distribution sets of synthetic data.
        N: number of shapes.
        M: number of parameters, e.g. metrics.
    iqr_stretch : float
        Factor to increase the filtering range: (1.5 + iqr_stretch) * iqr.
    metrices_to_filter : dictionary
        Contains all computed metrics (given in the infile), for every computed metric the item indicates if filtering
        applies on this parameter (no==0, yes==1).
    name : string
        String indicates which state, e.g. "dia" or "sys", is investigated. Substring is searched in syn_path folder
        structure.

    Returns
    -------
    None

    '''

    print("\n\nFilter synthetic data using real for metrics...\n")
    nr_filt = 0
    
    print("Synthetic dia shape was removed due to threshold :\n")
    for dirpath, subdirs, files in os.walk(syn_path):
        if name in dirpath and "leaflets" not in dirpath and \
            "orifice" not in dirpath and "intersect" not in dirpath and \
            "filter" not in dirpath:
            # get all .obj files in the directory
            files = sorted(glob.glob(dirpath + "\*.obj")) 
            
            # folder for filtered data
            filter_path = dirpath + "_filtered"
            if not os.path.exists(filter_path):
                os.makedirs(filter_path)

            # find outliers in synthetic data 
            outliers_idx = find_outliers(data_org, data_syn, metrices_to_filter, iqr_stretch=iqr_stretch)
            
            # create list with files to filter out
            # check if already filtered via indexing
            try:
              files_filter = list(files[i] for i in outliers_idx)
            except:
              # print("\n\n!!! Warning !!!\n\nData already filtered !")
              # print("\nRestore structure of synthetic shapes before filtering again.\n")
              # print("*** End of filtering ***\n")
              files_filter = []
            else:
              files_filter = []
              for idx in outliers_idx:
                  files_filter.append(files[idx])
                  ## alternative : list(files[i] for i in outliers_idx)
            
            # move files to filter in different folder
            for file in files_filter:
                shutil.move(file, filter_path)
                ##print(" - ", file.split('\\')[-1])
                nr_filt += 1
                    
    print(f"\n\nNumber of filtered synthetic meshes ({name}) :   ", nr_filt)  
    
    return



def detect_self_intersections(mesh, limit_self):
    '''
    Analyzing mesh and dectect self-intersections. 
    Possible to change threshold of intersections, either zero or max_value.

    Parameters
    ----------
    mesh : string
        Path to mesh .stl file to analyze.

    Returns
    -------
    result_dict : dictionary
        Result saved in dictionary, is mesh intersect free?

    '''
    
    # write path to file in specific path folder
    pymeshlab_folder = "pymeshlab"
    pymeshlab_file = "pymeshlab_file.txt"
    pathfile_pymeshlab = os.path.abspath(pymeshlab_folder + "/" + pymeshlab_file)

    with open(pathfile_pymeshlab, 'w') as f:
        f.write(mesh + "\n")
        f.write(str(limit_self))        
        
    # execute pymeshlab file to find self intersections
    output = subprocess.getoutput("python find_selfintersections.py")
    
    # save shell output in dictionary
    result_dict = ast.literal_eval(output)

    return result_dict


def compare_metrics(syn_metrics, thresholds, id_shape):
    '''
    Compares all given metric thresholds with the metrics of a synthetic shape. If one synthetic metric is
    out of the bounds of the given thresholds, set the filter flag to "True" for filtering.
    
    # TODO : need to adapt infile.py if this function is used in main.py !!!

    Parameters
    ----------
    syn_metrics : (N, M) numpy nd array
        Metrics of each synthetic shape.
        N: number of shapes.
        M: number of parameters, e.g. metrics.
    thresholds : (N,) nested list
        N metrics in the list, for each metric one sub-list contains lower [0] and upper [1] boundary of metric.
    id_shape : integer
        Id of synthetic shape.

    Returns
    -------
    filter_flag : bool
        Flag indicates if shape is filtered or not.

    '''
    
    filter_flag = False
    
    # check thresholds for each metric
    for j in range(len(syn_metrics[0,:])):
        if thresholds[j] == None:
            continue
        value = syn_metrics[id_shape, j]
        if value < min(thresholds[j]) or value > max(thresholds[j]):
            filter_flag = True
            break
        
    return filter_flag