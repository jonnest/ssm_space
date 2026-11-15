################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import os
import time
import sys
import numpy as np
import pyvista as pv

from ssm.ssm_mv_tomtec.metrics import align_to_commissure
from tools.pointclouds import getOrderedBoundaryCurve
from ssm.ssm_general.ssm_linear import calculateSSM
from tools.io_utils import readTriangularSurfaceMesh, writeOBJ, \
                            reshapeFacesToPolyData
                            
                            
def get_mesh_list(data_path, use_cases, name):
    # initialize list
    mesh_list = []

    # walk through folder structure
    for dirpath, subdirs, files in os.walk(data_path):
        # check if cases in use_cases are part of dirpath
        c = 0
        for i in list(map(str, use_cases)):
            if (dirpath.find(i) != -1):
                c += 1
        if (c >= 1):

            for file in files:
                # check if systolic or diastolic state
                if f"{name}" in file and "aligned" in file:
                    # mesh file of loop
                    mv_mesh = os.path.join(dirpath, file)

                    # add mesh to list
                    mesh_list.append(mv_mesh)

    return mesh_list


def read_aligned_data(nr_cases, data_path, use_cases, init_mesh_file, name):
    '''
    Read the aligned original data from the data_path structure.

    Parameters
    ----------
    nr_cases : integer
        Number of cases to analyse, for initialization.
    data_path : string
        Absolute path to the original data structure directory.
    use_cases : (M,) numpy array
        Array of integers with the case names to analyze.
    init_mesh_file : string
        Absolute path to a initialization file for the number of vertices and faces.
    name : string
        Name string written in the files to be saved.

    Returns
    -------
    S : (N, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.

    '''
    # Initialize metrics matrix
    v, f = readTriangularSurfaceMesh(init_mesh_file)
    S = np.empty((len(v)*3, nr_cases))
    
    print("\nNumber of vertices ", len(v))
    print(f"Read aligned data for {name} SSM construction...")
    
    # counters for dia and sys files
    nr_shapes = 0
    # walk through folder structure
    for dirpath, subdirs, files in os.walk(data_path):
        # check if cases in use_cases are part of dirpath
        c = 0
        for i in list(map(str, use_cases)):
            if(dirpath.find(i)!=-1):
                c += 1
        if(c >= 1):
            
            for file in files:
                # check if systolic or diastolic state
                if f"{name}" in file and "aligned" in file:
                    # mesh file of loop
                    mv_mesh = os.path.join(dirpath, file)
    
                    # read mesh
                    v, _ = readTriangularSurfaceMesh(mv_mesh)
                    v_1d = np.array(np.reshape(v, (len(v)*3, 1)))
                    
                    # check for non-fitting vertice number
                    if len(v)*3 != len(S):
                        print("\nFile has bad node connection\n", dirpath)
                        continue
                    
                    S[:,nr_shapes] = v_1d[:,0]
                    nr_shapes += 1
    
    
    return S


def ssm_setup(S, ssm_path, print_variance, name):
    '''
    Conduct the linear SSM on the shape matrix "S" and save the mean, modes and
    variances in the SSM folder as .txt files.

    Parameters
    ----------
    S : (N, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    ssm_path : string
        Absolute path to the SSM directory to save the txt files.
    print_variance : float
        While computing, print the modes' variances in the console, cumulative variance of X percent.
    name : string
        Name string written in the files to be saved.

    Returns
    -------
    None.

    '''
    print(f"\nSSM {name} setup...")
    # Compute linear SSM via SVD
    t1 = time.time()
    mean, modes, var = calculateSSM(S)
    t2 = time.time()

    
    print("\nNumber of modes for diastolic shapes  ", len(var))
    print(f"\nTime snapshot SSM MV-{name}  ", round(t2-t1,5)," seconds")
    
    print(f"\n{name} state")
    for i in range(len(S[0,:])-1):
        print(f"Relative part of {i+1}. mode ", round(var[i]/np.sum(var)*100,2), "%")
        print(f"Cumulative varianz for {i+1} modes     ", round(np.sum(var[:i+1])/np.sum(var)*100,2), "%")
        if np.sum(var[:i+1])/np.sum(var) > print_variance:
            break

    # Save SSM results
    print(f"\nSave SSM {name} of real data...")
    
    f_mean, f_modes, f_var = f"{name}_mean.txt", f"{name}_modes.txt", f"{name}_var.txt"

    if not os.path.exists(ssm_path):
        os.makedirs(ssm_path)
        print("\nSSM directory in project folder created!\n")
    
    files = [f_mean, f_modes, f_var]

    ssm = [mean, modes, var]

    # save for dia
    for i, file in enumerate(files):
        file_path = os.path.abspath(ssm_path + "/" + file)
        # write to txt files
        np.savetxt(file_path, ssm[i])

    return


def align_real_data(data_path, use_cases, parameters, alignment, name):
    '''
    Align the mitral valve according the "alignment" method, either along 
    landmarks or via the iterative closest point (ICP) algorithm (not implemented yet). 

    Parameters
    ----------
    data_path : string
        Absolute path to the original data structure directory.
    use_cases : (M,) numpy array
        Array of integers with the case names to analyze.
    parameters : dict
        Dictionary containg all relevent parameters of the project, mainly arrays of id's.
    alignment : string
        Alignment either "landmark" or "icp", sets the method used for alignment.
    name : string
        Name string written in the files to be saved.

    Returns
    -------
    None.

    '''
    # determined vertex id's
    # depends on definition of vertex id structure
    lat_com = parameters["lat_com"]
    med_com = parameters["med_com"]
    v_hori = parameters["v_hori"]
    bnd_id_annulus = parameters["bnd_id_annulus"]
    
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
                if file == f"MV_{name}.obj":
                    # mesh file of loop
                    mv_mesh = os.path.join(dirpath, file)
    
                    v, f = readTriangularSurfaceMesh(mv_mesh)
                    # get boundary curve, check number of boundaries
                    v_bn, f_bn = getOrderedBoundaryCurve(v, f, 2)
                    # here : assumption that annulus is always first list item, orifice second item
                    annulus = v_bn[bnd_id_annulus]
                    
                    # choose alignment method
                    if alignment == "landmark":
                        align_to_commissure(v, f, annulus, lat_com, med_com, v_hori, mv_mesh, save_geo=True)
                    
                    elif alignment == "icp":
                        # TODO : ICP
                        pass
                    
    return



def plot_input_data(S, init_mesh_file, use_cases, rows, colns):
    '''
    Visualization of a sample of the synthetic shapes.

    Parameters
    ----------
    S_shapes : (N, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    init_mesh_file : string
        Absolute path to the initialization file for the number of vertices and faces.

    Returns
    -------
    None.

    '''
    print("\nVisualization of original geometries in pyvista\nClose window to continue!")
    # initialize
    v, f = readTriangularSurfaceMesh(init_mesh_file)
    
    # visualize sample
    pv.global_theme.show_edges = True
    pv.global_theme.background = 'white'

    pl = pv.Plotter(shape=(rows,colns))
    k = 0
    for i in range(rows):
        for j in range(colns):
            
            mode = S[:,k].reshape(int(len(S[:,k])/3), 3)
            faces_pv = reshapeFacesToPolyData(f)
            lin_mode = pv.PolyData(mode, faces_pv.flatten().astype(int))
        
            pl.subplot(i, j)
            pl.add_mesh(lin_mode)
            pl.view_xy()
            # pl.camera.roll += 180
            pl.camera.roll -= 90
            pl.camera.zoom(1)
            
            txt = str(use_cases[k])
            pl.add_text(txt, position='upper_left', color='black', font_size=8)
            k += 1
            if k==len(S[0,:]):
                break
        if k==len(S[0,:]):
            break
    
    pl.link_views()
    pl.show()
    
    return 