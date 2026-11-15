################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
from funcs.pointclouds import *
from funcs.meshes import *
from funcs.pymeshlab import *

import scipy as scp

# procrustes scikit libraries
# scipy.linalg.orthogonal_procrustes
# scipy.spatial.procrustes


def procrustes_ortho_multi(data_path, nr_it):
    '''
    Procrustes Alignment without scaling. Only orthogonal matrices are used, 
    as rotation and reflection transformation matrices.

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    nr_it : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    save_folder = "procrustes_ortho_multi"
    save_path = os.path.join("." + "/" + save_folder)

    ###############################################################################
    # Check whether the specified path exists or not
    isExist = os.path.exists(save_folder)
    if not isExist:
        os.makedirs(save_folder)

    ###############################################################################
    # do multiple iterations of mapping and computing of mean shape
    os.chdir(data_path)
    cases = os.listdir()

    v1, f = readTriangularSurfaceMesh(cases[0])
    # get all shapes for matrix size
    v_all = np.empty((len(v1) * 3, len(cases)))

    for j in range(nr_it):
        # save all shapes in matrix
        for i in range(len(cases)):
            v, f = readTriangularSurfaceMesh(cases[i])
            v_1d = np.array(np.reshape(v, (len(v1)*3, 1)))
            v_all[:, i] = v_1d[:, 0]

        # compute mean s
        v_mean = v_all.mean(axis=1)
        v_mean = np.array(np.reshape(v, (len(v1), 3)))
        # write mean shape vertices
        writeTriangularSurfaceMesh(v_mean, f, os.path.join(save_mean, f"mean_{j}.stl"))

        # here
        # v1 : to be mapped
        # v2 : target matrix
        # minimizes (A @ R) - B, subject to R.T @ R = I

        save_path_it = save_path + f"_{j}"
        # Check whether the specified path exists or not
        isExist = os.path.exists(save_path_it)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path_it)

        for i in range(len(cases)):
            # read mesh
            v, _ = readTriangularSurfaceMesh(cases[i])
            # translate to center
            ctr = v.mean(axis=0)
            v = v - ctr
            # get orthogonal procrustes transformation
            R, scale = scp.linalg.orthogonal_procrustes(v, v_mean)
            # map v1 to v2
            v_map = np.matmul(v, R)
            save_file = os.path.join(save_path_it, cases[i])
            # write mapped file
            writeTriangularSurfaceMesh(v_map, f, save_file[:-3] + "stl")

        # update data path
        os.chdir(save_path_it)
        cases = os.listdir()

    print("\nNumber of alignment iterations : ", j+1)
    
    return
