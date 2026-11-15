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
import random
import numpy as np
import pyvista as pv

from ssm.ssm_general.shape_synthesis import shape_reconstruct
from tools.io_utils import readTriangularSurfaceMesh, writeOBJ, \
                            reshapeFacesToPolyData
                

def create_synthetic_cohort(ssm_path, syn_path, plot_path, init_mesh_files, 
                            nr_shapes, sigma, var_threshold, seed_value, shape_parameter,
                            S, nr_digits, nr_shape_param, nr_scalar, use_demos,
                            include_gender,
                            properties_through_shape, 
                            properties_through_parameter, S_demo, name, states=None):
    '''
    Create synthetic cohort from real data. Choose the method how the shape
    parameters are calculated and superposed. 

    Parameters
    ----------
    ssm_path : string
        Absolute path to SSM setup files containing mean geometry, variance and the mode matrix.
    syn_path : string
        Absolute path to synthetic shape directory.
    plot_path : string
        Absolute path to save plots of the workspace.
    init_mesh_files : list of strings
        Path to mesh file for initialization of the shape matrices and faces.
    nr_shapes : integer
        Number of synthetic shapes to generate.
    sigma : float
        Standard deviation of the Gaussian Normal distribution for the shape parameter selection.
    var_threshold : float
        Relative variance threshold determines how much variance percentage of the modes is used.
        Value is part of the interval [0,1].
    seed_value : integer or else
        Random value for seed of random generator. For reproducable "randomly" generated 
        shapes, re-use the seed value. If no integer, seed_value is generated randomly. 
    shape_parameter : string
        Method selection, how to generate synthetic shapes, currently "mean", "eigenvalues", "copula".
    S : (N, M) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        M number of different shapes.
    nr_shape_param : integer
        For copula synthesis : number of mode parameters to plot in the visualization.
    nr_scalar : integer
        Number of scalar values included in SSM.
    name : string
        Name of the state, e.g. "dia" or "sys".
    states : list
        List of state names to process, e.g. ["dia", "sys"]

    Returns
    -------
    shapes : (N, nr_shapes) numpy nd array
        Shape matrix with columns as coordinates of each shape.
        N number of vertices.
        nr_shapes : number of different synthetic shapes.
    seed_value : integer
        Random value for seed of random generator. For reproducable "randomly" generated 
        shapes, re-use the seed value..

    '''
    # read ssm setup   
    f_mean, f_modes, f_var = f"{name}_mean.txt", f"{name}_modes.txt", f"{name}_var.txt"
    mean  = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_mean))
    modes = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_modes))
    var   = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_var))
    # get eigenvalues from SSM variance
    ev = np.sqrt(abs(var))
    
    # create a seed for random seed value
    if type(seed_value) != int:
        seed_value = random.randrange(2**32)
    # save this seed somewhere. So if the result is good, use this seed to reproduce it
    print("\nSeed value: ", seed_value, "\n")
    print(f"\nCreate synthetic {name} cohort and save geometries...")
    
    t4 = time.time()
    # generate synthetic shapes
    shapes = shape_reconstruct(mean, ev, modes, nr_shapes,
                                sigma, var_threshold, plot_path, ssm_path, 
                                properties_through_shape, 
                                properties_through_parameter, S_demo, nr_scalar, use_demos, include_gender,
                                shape_parameter, seed_value=seed_value,
                                S_real=S, nr_shape_param=nr_shape_param, name=name)
    t5 = time.time()
        
    # if single state, list of states is "None", separate SSM
    if states == None:
        # if not multiple states to be processed
        # then states equals the single "name" as a list
        states = [name]
    
    # initialization, read all vertices and faces of all files
    v_all = [readTriangularSurfaceMesh(file)[0] for file in init_mesh_files]
    f_all = [readTriangularSurfaceMesh(file)[1] for file in init_mesh_files]

    # get vertice length of every states
    # number of rows times 3 coordinate directions
    v_len = [len(v_all[i])*3 for i in range(len(v_all))]
    
    # get indices of all states
    # start and end indices for row of shape matrix to find shape vertices
    v_idx = [sum(v_len[:i+1]) for i in range(len(v_len))]
    # add zero index to list
    v_idx.insert(0, 0)
    
    for j, state in enumerate(states):
        save_folder = os.path.join(syn_path, f"{state}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
                
        # save new geometries, loop over directory
        for dirpath, subdirs, files in os.walk(syn_path):
            if f"{state}" in dirpath:
                for file in files:
                    os.remove(os.path.join(dirpath, file))
        
        for i in range(nr_shapes):
            # select vertices from shape matrix - check if scalar properties are in shape matrix
            if properties_through_shape == True:
                new_1d = shapes[:-nr_scalar,i]
            else:
                new_1d = shapes[v_idx[j]:v_idx[j+1], i]

            # reshape 1d vertices to nd array
            new_v = np.array(np.reshape(new_1d, (int(len(new_1d)/3), 3)))
            # fill counter with zeros
            nr = str(i + 1).zfill(nr_digits) 
            save_file = os.path.join(syn_path, f"{state}", f"shape_{nr}.obj")
            
            # generate first comment line of .OBJ file
            first_line = ""
            for s in range(nr_scalar):
                first_line = first_line +  f"{shapes[-nr_scalar+s,i]}" + ","
            
            # save geometry as .OBJ file
            writeOBJ(save_file, new_v, f_all[j], first_line)
            
        t6 = time.time()
        
        print("\nTime : generate synthetic shape matrices  ", round(t5-t4, 5)," seconds")
        print("Time : save synthetic shapes              ", round(t6-t5,5)," seconds")
        print("\n")
    
    return shapes, seed_value
    


def plot_synthetic_sample(S_shapes, init_mesh_file, nr_plot_synthetic):
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
    nr_plot_synthetic : integer
        Number of subplots in one figure in the visualization.

    Returns
    -------
    None.

    '''
    print("\nVisualization of synthetic geometries in pyvista\nClose window to continue!")
    
    # initialize
    v, f = readTriangularSurfaceMesh(init_mesh_file)
    
    # visualize sample
    pv.global_theme.show_edges = True
    pv.global_theme.background = 'white'
    
    # compute displaying relation for screen 4:3 - 4*a:3*a
    a = np.sqrt(nr_plot_synthetic/12)
    rows = int(3*a)
    colns = int(4*a)
    n_plots = rows*colns

    pl = pv.Plotter(shape=(rows,colns))
    k = 0
    for i in range(rows):
        for j in range(colns):
            
            mode = S_shapes[:,k].reshape(int(len(S_shapes[:,k])/3), 3)
            faces_pv = reshapeFacesToPolyData(f)
            lin_mode = pv.PolyData(mode, faces_pv.flatten().astype(int))
        
            pl.subplot(i, j)
            pl.add_mesh(lin_mode)
            pl.view_xz()
            pl.camera.roll += 180
            pl.camera.zoom(1)
            #pl.add_text(txt, position='upper_left', color='black', font_size=8)
            k += 1
    
    pl.link_views()
    pl.show()
    
    return 

