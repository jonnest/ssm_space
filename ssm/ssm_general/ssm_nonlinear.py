################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
from numpy.linalg import norm
import statistics
import time
import os
import random

from tools.io_utils import *
from tools.pointclouds import getOrderedBoundaryCurve, createCoordinateSystem, createRightHandSystem, \
                                alignCoordinateSystems

from morphomatics.geom import Surface
from morphomatics.stats import StatisticalShapeModel
from morphomatics.manifold import FundamentalCoords


def nonlinear_synthesis(mesh_list, syn_path, nr_shapes, sigma, var_threshold, 
                        seed_value, nr_digits, print_variance, parameters, state):
    # determined vertex id's
    # depends on definition of vertex id structure
    lat_com = parameters["lat_com"]
    med_com = parameters["med_com"]
    v_hori = parameters["v_hori"]
    bnd_id_annulus = parameters["bnd_id_annulus"]

    # save geometries
    save_folder = os.path.join(syn_path, f"{state}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save new geometries, loop over directory
    for dirpath, subdirs, files in os.walk(syn_path):
        if f"{state}" in dirpath:
            for file in files:
                os.remove(os.path.join(dirpath, file))

    # create a seed for random seed value
    if type(seed_value) != int:
        seed_value = random.randrange(2**32)
    # save this seed somewhere. So if the result is good, use this seed to reproduce it
    print("\nSeed value: ", seed_value, "\n")

    # read meshes
    meshes = []
    for mesh in mesh_list:
        meshes.append(pv.read(mesh))
    
    # NOTE for reshaping: first column of faces is filled with int 3 - can be cut out
    as_surface = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    surfaces_dia = [as_surface(m) for m in meshes]
    
    # construct the nonlinear SSM
    SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
    SSM.construct(surfaces_dia)
    
    # standard deviation associated to the modes; std_dev is sqrt of variance
    var = SSM.variances
    std = np.sqrt(SSM.variances)
    
    # print variances and cumulative variance
    for i in range(len(std)):
        print(f"Relative part of {i+1}. mode         ",  round(var[i]/np.sum(var)*100,2), " %")
        print(f"Cumulative variance for {i+1} modes  ",  round(np.sum(var[:i+1])/np.sum(var)*100,2) , " %")
        if np.sum(var[:i+1])/np.sum(var) > print_variance:
            break

    # synthetic shapes
    # initialize metrics matrix
    nr_coord = len(SSM.space.from_coords(SSM.modes[0]))*3
    S = np.empty((nr_coord, nr_shapes))
    
    # number of coordinates in shape space
    nr_coord_space = len(SSM.modes[0])

    # initialize length of nr_modes
    nr_modes = len(std)

    # get faces
    faces = readTriangularSurfaceMesh(mesh_list[0])[1]

    for j in range(nr_shapes):
        # initialize shape matrix for one synthetic case
        shape = np.empty((nr_coord_space, nr_modes))
        
        for m in range(nr_modes):
            # select method of reconstruction
            # normal distribution around mean value with stdev sigma
            b = np.random.normal(loc=0, scale = sigma * std[m])
            # shape excitation
            shape[:, m]  = b * SSM.modes[m]
        
        # combine mean coordinates and sum of randomized shape variations together
        shape_var = SSM.space.exp(SSM.mean_coords, np.sum(shape, axis=1)) 
        # back transformation to linear / euclidean space
        shape_var_nd = SSM.space.from_coords(shape_var)
        # shape_var_1d = np.array(np.reshape(shape_var_nd, (-1, 1)))
        shape_var_1d = shape_var_nd.reshape(-1)
        # add synthetic mode to shape matrix
        S[:,j] = shape_var_1d

        ### aligning the synthetic geometry
        # best fit plane of annulus using SVD of point cloud - rotate into plane
        v_bn, _  = getOrderedBoundaryCurve(shape_var_nd, faces,2)
        annulus = v_bn[bnd_id_annulus]

        [a1, a2, a3], ctr_annulus = createCoordinateSystem(np.array(annulus))

        # realign the vertices to have equal orientation for all possible original orientations
        # check if geometry is already aligned, so that normal vector (a3) of SVD is already equal to [0,0,1] base vector
        e1 = shape_var_nd[med_com, :] - shape_var_nd[lat_com, :]
        e2 = shape_var_nd[v_hori, :] - ctr_annulus
        e3 = np.cross(e1, e2)

        ctr = shape_var_nd.mean(axis=0)
        base_target = np.identity(3)
        origin = [0.0, 0.0, 0.0]
        v_com = shape_var_nd[lat_com, :] - shape_var_nd[med_com, :]

        # align depending on normal vector
        if np.dot(e3, a3) > 0:
            # use negative base vector for orientation
            base_com = createRightHandSystem(v_com, -a3)
        else:
            # use positive base vector for orientation
            base_com = createRightHandSystem(v_com, a3)

        # aligned vertices along commissures and in annulus plane
        v_com_plane, _, _, _ = alignCoordinateSystems(shape_var_nd, ctr, base_com, origin, base_target)

        ### Save geometry
        # fill counter with zeros
        nr = str(j + 1).zfill(nr_digits)
        save_file = os.path.join(syn_path, f"{state}", f"shape_{nr}.obj")
        # save geometry as .OBJ file
        writeOBJ(save_file, v_com_plane, faces)

    return S

    