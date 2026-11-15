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
import time
import pandas as pd

from ssm.ssm_mv_tomtec.metrics import getMetricsMain, read_scalars_from_obj
from tools.io_utils import readTriangularSurfaceMesh
from filter_data import detect_anterior_artefact

def get_real_data_metrics(nr_cases, nr_props, use_metrics, metrics_caps, use_cases, 
                          data_path, project_path, parameters, metric_path, name, 
                          remove_artefact=False, *args):
    '''
    Get the metrics defined in "use_metrics" from the original data set and save 
    in a metrics file in the metrics folder.

    Parameters
    ----------
    nr_cases : integer
        Number of cases to analyse, for initialization.
    nr_props : integer
        Number of properties/metrics to analyze.
    use_metrics : dict
        Dictionary containing the names of the metrics to analyze.
    metrics_caps : list
        List with strings of the metric captions.
    use_cases : (M,) numpy array
        Array of integers with the case names to analyze.
    data_path : string
        Absolute path to the original data structure directory.
    project_path : string 
        Absolute path to the project directory containing all subfolders.
    parameters : dict
        Dictionary containg all relevent parameters of the project, mainly arrays of id's.
    metric_path : string
        Absolute path to the subfolder for saving the metric files.
    name : string
        Name string written in the files to be saved.
    *args : optional arguments
        Here: start, width, rows, nr_v_circ, ang_threshold for the search window to analyze
        curvature in the mesh.

    Returns
    -------
    None.

    '''
    
    print(f"\nGet metrics of real {name} data...")
    # initialize metrics matrix
    metrics = np.empty((nr_cases, nr_props))
    
    # counters for files
    nr_shapes = 0
    max_angles = []
    
    # variable names
    var_names = ["start", "width", "rows", "nr_v_circ", "ang_threshold"]
    # Assign the optional arguments to the variable names
    var = dict(zip(var_names, args))
    
    # walk through folder structure
    # get metrics for every bottom level file starting with "MV_"
    for dirpath, subdirs, files in os.walk(data_path):
        # check if cases in use_cases are part of dirpath
        c = 0
        for i in list(map(str, use_cases)):
            if(dirpath.find(i)!=-1):
                c += 1
        if(c >= 1):
            for file in files:
                # name is either 'dia' or 'sys'
                if file == f"MV_{name}.obj":
                    # mesh file of loop
                    mv_mesh = os.path.join(dirpath, file)
                    
                    # get metrics dictionary
                    metrics_dict = getMetricsMain(mv_mesh, project_path, 
                                                  use_metrics, parameters)
                    
                    # write metrics to file
                    metric_file = os.path.join(dirpath, "metrics.txt")
    
                    with open(metric_file, 'w') as m:
                        m.write(str(metrics_dict))

                    # loop over values in dictonary
                    for count, items in enumerate(metrics_dict.items()):
                        metrics[nr_shapes, count] = items[1]
                    nr_shapes += 1
                    
                    if remove_artefact == True:
                        # read mesh
                        v, f = readTriangularSurfaceMesh(mv_mesh)
                        # compute angles for artefact
                        _, max_angle = detect_anterior_artefact(v, f, var["start"],
                                                                var["width"], var["rows"], 
                                                                var["nr_v_circ"], var["ang_threshold"])
                        max_angles.append(max_angle)

    if remove_artefact == True:
        max_angles = np.array(max_angles)
        print("\norg: mean max angles  ", round(max_angles.mean(),2))
        print("\norg: std max angles   ", round(max_angles.std(),2))
    
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    # write all metrics in txt file
    all_metrics_file = os.path.join(metric_path, f"metrics_{name}_original.txt")
    np.savetxt(all_metrics_file, metrics, header="\t".join(metrics_caps))
    # save csv via pandas
    df_metrics = pd.DataFrame(metrics, columns = metrics_caps)
    df_metrics.to_csv(os.path.join(metric_path, f"metrics_{name}_original.csv"))

    return



def get_synthetic_data_metrics(nr_shapes, nr_props, use_metrics, metrics_caps, syn_path,
                              data_path, project_path, parameters, metric_path, 
                              nr_scalar, name, ext="", remove_artefact=False, *args):
    '''
    Get the metrics defined in "use_metrics" from the synthetic data set and save 
    in a metrics file in the metrics folder.

    Parameters
    ----------
    nr_shapes : integer
        Number of cases to analyse, for initialization.
    nr_props : integer
        Number of properties/metrics to analyze.
    use_metrics : dict
        Dictionary containing the names of the metrics to analyze.
    metrics_caps : list
        List with strings of the metric captions.
    syn_path : string
        Absolute path to the synthetic data structure directory.
    data_path : string
        Absolute path to the original data structure directory.
    project_path : string 
        Absolute path to the project directory containing all subfolders.
    parameters : dict
        Dictionary containg all relevent parameters of the project, mainly arrays of id's.
    metric_path : string
        Absolute path to the subfolder for saving the metric files.
    nr_scalar : integer
        Number of scalars included at the bottom of the shape matrix .
    name : string
        Name string written in the files to be saved.
    ext : string, optional
        Extension string of name to save metrics. The default is "".
    *args : optional arguments
        Here: start, width, rows, nr_v_circ, ang_threshold for the search window to analyze
        curvature in the mesh.
        
    Returns
    -------
    None.

    '''
    
    print(f"\nGet metrics of synthetic {name} data...")
    ### initialize metrics matrix
    # metrics = np.empty((nr_shapes, nr_props + nr_scalar))
    metrics = np.zeros((nr_shapes, nr_props))
    
    print("\nMetrics shape   ", metrics.shape)

    t1 = time.time()
    # counters for dia and sys files
    nr_shapes = 0
    max_angles = []
    
    # variable names
    var_names = ["start", "width", "rows", "nr_v_circ", "ang_threshold"]
    # Assign the optional arguments to the variable names
    var = dict(zip(var_names, args))
    
    # TODO : nr_scalar if condition for properties_through_shape ? 
    # or overwrite the metrics with the copula scalars
    
    # walk through folder structure
    # get metrics for every bottom level file starting with "MV_"
    for dirpath, subdirs, files in os.walk(syn_path):
        for file in files:
            if "shape" in file and f"{name}" in dirpath and "anterior" not in file and \
                "posterior" not in file and "orifice" not in file and \
                "filtered" not in dirpath and "artefact" not in dirpath and "intersect" not in dirpath:
                # mesh file of loop
                mv_mesh = os.path.join(dirpath, file)
                
                # get metrics dictionary
                metrics_dict = getMetricsMain(mv_mesh, project_path, use_metrics, parameters)

                # read scalars from .OBJ file
                if nr_scalar != 0:
                    scalar_array = read_scalars_from_obj(mv_mesh)
                    
                # write metrics to file
                metric_file = os.path.join(dirpath, "metrics.txt")

                # loop over values in dictonary
                for count, items in enumerate(metrics_dict.items()):
                    metrics[nr_shapes, count] = items[1]
                
                ### if scalar values shall be added to metrics
                # for j in range(nr_scalar):
                #     metrics[nr_shapes, count+1+j] = scalar_array[j]
                
                nr_shapes += 1
                
                if remove_artefact == True:
                    # read mesh
                    v, f = readTriangularSurfaceMesh(mv_mesh)
                    # compute angles for artefact
                    _, max_angle = detect_anterior_artefact(v, f, var["start"],
                                                            var["width"], var["rows"], 
                                                            var["nr_v_circ"], var["ang_threshold"] )
                    max_angles.append(max_angle)

    # remove rows having all zeroes
    metrics = metrics[~np.all(metrics == 0, axis=1)]

    if remove_artefact == True:
        max_angles = np.array(max_angles)
        print("\norg: mean max angles  ", round(max_angles.mean(),2))
        print("org: std max angles   ", round(max_angles.std(),2))

    t2 = time.time()
    print("\nTime : loop over ", nr_shapes, f" {name} shapes : ", round(t2-t1,2) ," seconds")
    
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    # write all metrics in txt file
    all_metrics_file = os.path.join(metric_path, f"metrics_{name}_synthetic{ext}.txt")
    np.savetxt(all_metrics_file, metrics, header="\t".join(metrics_caps))
    # save csv via pandas
    df_metrics = pd.DataFrame(metrics, columns = metrics_caps)
    df_metrics.to_csv(os.path.join(metric_path, f"metrics_{name}_synthetic{ext}.csv"))
    
    return