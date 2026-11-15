################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import seaborn as sns
import sys
import pyvista as pv
import scipy as scp

os.chdir("..")

# from ssm.ssm_general.shape_synthesis import *
# from tools.io_utils import *
# from ssm.ssm_mv_tomtec.infile import * 

'''
Infile for Statistical Shape Analysis for analyzing Mitral Valve data from
TomTec. 

Only change parameters, paths and settings in this file.

Select which subroutines in the main.py file are executed.

'''

import numpy as np

###############################################################################
# general settings
###############################################################################
# number of synthetic shapes
nr_shapes = 2000

# workspace is name main folder
workspace = f"workspace_n{nr_shapes}"

# paths and excel files to select cases and tomtec data
workpath = "C:/Users/user/Documents/10_MV_SSM"
##workpath = "/home/user/Documents/mv_ssm_loc/MV_SSM/"
case_file = "CasesOverview_pre.xlsx"
tomtec_file = "TomTec-Results.csv"

# metrics to evaluate, no==0, yes==1
circ_annulus_flag = 1
A_annulus_flag = 1
h_annulus_flag = 1
ratio_annulus_flag = 1
ci_annulus_flag = 1
circ_orifice_flag = 1
A_orifice_orif_flag = 0
A_orifice_annu_flag = 0
h_orifice_flag = 1
ratio_orifice_flag = 1
ci_orifice_flag = 1
len_lf_ant_flag = 1
len_lf_post_flag = 1
ang_lf_ant_flag = 1
ang_lf_post_flag = 1
A_lf_ant_flag = 1
A_lf_post_flag = 1
A_3d_orifice_flag = 1



###############################################################################

# folder containing real data
case_name = "OBJ"
# folder to save metrics as txt files
save_name = "metrics"
# folder to save ssm results
ssm_name = "ssm"
# folder to save synthetic geometries
syn_name = "synthetic"
# folder to save all plots
test_name = "test"
# folder to save all statistical evaluations
stats_name = "stats"

# paths resulting from workpath with respective folders
project_path = os.path.abspath(workpath + "/" + workspace + "/")
# subdirectoreis of project path
data_path = os.path.abspath(project_path +  "/" + case_name + "/")
metric_path = os.path.abspath(project_path +  "/" + save_name + "/")
ssm_path = os.path.abspath(project_path +  "/" + ssm_name + "/")
syn_path = os.path.abspath(project_path +  "/" + syn_name + "/")
test_path = os.path.abspath(project_path +  "/" + test_name + "/")
stats_path = os.path.abspath(project_path +  "/" + stats_name + "/")
# path to selection file of cases
select_path = os.path.abspath(project_path +  "/" + case_file + "/")

# select real cases to use from case_file
cases = pd.read_excel(select_path)
nr_cases = len(cases[cases["use"]==1])
use_cases = cases.loc[cases["use"] == 1]
use_cases = use_cases["case"].to_numpy()

print("\nUse    ", len(use_cases), " cases of real data")
print("\nCreate ", nr_shapes, " synthetic cases\n")

# tomtec evalutation file
tomtec_path = os.path.abspath(project_path + "/" + tomtec_file)

# initialization file for metric matrices
dia_init_mesh_file = os.path.abspath(data_path + "/" + str(use_cases[0]) +"/middia/MV_middia.obj")
sys_init_mesh_file = os.path.abspath(data_path + "/" + str(use_cases[0]) +"/midsys/MV_midsys.obj")

# number of shape parameters of real data to analyze
nr_shape_param = 10

###############################################################################

# read shape parameters
dia_org_file = os.path.join(test_path, "b_real_dia.txt")
dia_syn_file = os.path.join(test_path, "b_copula_dia.txt")

dia_org = np.loadtxt(dia_org_file)
dia_syn = np.loadtxt(dia_syn_file)

###############################################################################


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



def plot_real_data_shape_parameters(b_real, b_copula, nr_shape_param, plot_path, name, *sample_arg):
    
    start = 0
    
    # if argument
    b_samples = sample_arg[0]
    start = sample_arg[1]
    color = plt.cm.rainbow(np.linspace(0, 1, len(b_samples)))
    
    top=0.95
    bottom=0.055
    left=0.05
    right=0.98
    hspace=0.25
    wspace=0.8
    # fonts
    font = {'size' : 8}
    
    fig = plt.figure(figsize=(11,7))
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
        hspace=hspace, wspace=wspace)
    
    # parameters for histogram
    kwargs = dict(histtype='step', alpha=0.9, density=True)

    k = 0
    for i in range(start, nr_shape_param + start):
        
        ax = plt.subplot(2, int(np.ceil(nr_shape_param/2)), k + 1)  
        
        # shapiro test for normal distribution
        _, p_real = scp.stats.shapiro(b_real[:,i])
        _, p_copula = scp.stats.shapiro(b_copula[:,i])
        txt1 = "p_real=" + str(round(p_real,2))
        txt2 = "p_cop=" + str(round(p_copula,2))

        
        # Creating plot
        plt.hist(b_real[:,i], label=txt1, **kwargs)
        plt.hist(b_copula[:,i], label=txt2, **kwargs)
        
        # plot vertical lines
        for j in range(len(b_samples)):
            plt.axvline(b_test[j,i], color=color[j], linestyle='dashed', linewidth=1)
        
        ax.set_title(f"mode {i+1}")
        ax.legend()
        
        # # shapiro test for normal distribution
        # _, p_real = scp.stats.shapiro(b_real[:,i])
        # _, p_copula = scp.stats.shapiro(b_copula[:,i])
        # txt = ["p_real=" + str(round(p_real,2)), "p_cop=" + str(round(p_copula,2))]
        # ax.legend(txt)
        # print(f"\ntxt     mode {i+1}  \n", txt)
        
        k += 1
        
    fig.tight_layout()
    plt.show()
    
    plt.legend([f"{name} real", f"{name} syn"], bbox_to_anchor=(2.3, 0.5), ncol=1)
    shape_param_file = os.path.abspath(plot_path + f"/shape_param_{name}.png" )
    plt.savefig(shape_param_file, dpi=300)
    
    '''
    print("\nPrint shapiro test for eigenvalues of real data:")
    for i in range(len(b_real[0,:])):
        # shapiro test for normal distribution
        stats, p_shap = scp.stats.shapiro(b_dia[:,i])
        if p_shap <= 0.05:
            print(f"Mode {i+1} ", round(p_shap,4)) 
    '''        
            
    return


###############################################################################
# read test shapes 
##test_shapes = [5,15,18,32,51,55,70,77,119,125,132,158,159,165]
test_shapes = [51]

# io shapes
##test_shapes = [ 1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 14]
##test_shapes = np.arange(80, 115)

# Initialize metrics matrix
v, f = readTriangularSurfaceMesh(dia_init_mesh_file)

S_test = np.empty((len(v)*3, len(test_shapes)))
# counters for dia and sys files
nr_shapes = 0
for i in test_shapes:
    
    name = "shape_" + str(i).zfill(5) + ".obj"
    # mesh file of loop
    mv_mesh = os.path.abspath(syn_path +  "\\middia\\" + name)
    
    # read mesh
    v, _ = readTriangularSurfaceMesh(mv_mesh)
    v_1d = np.array(np.reshape(v, (len(v)*3, 1)))
    
    # check for non-fitting vertice number
    if len(v)*3 != len(S_test):
        print("\nFile has bad node connection\n")
        continue
    
    S_test[:,nr_shapes] = v_1d[:,0]
    nr_shapes += 1
    

###############################################################################
# read SSM
name = "dia"

# read ssm setup   
f_mean, f_modes, f_var = f"{name}_mean.txt", f"{name}_modes.txt", f"{name}_var.txt"

mean  = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_mean))
modes = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_modes))
var   = np.loadtxt(os.path.abspath(ssm_path+ "/" + f_var))

# compute test shape parameters
b_test = np.empty(( len(S_test[0,:]), len(modes[0,:]) )) 
                  
# loop over all shapes in shape matrix S
for i in range(len(S_test[0,:])):
    # least squares solution of linear matrix system
    b_i, residual, _, _ = np.linalg.lstsq(modes, S_test[:,i])
    
    b_test[i,:] = b_i


for i in range(9):
    name = "dia_" + f"{i*10}"    
    plot_real_data_shape_parameters(dia_org, dia_syn, nr_shape_param, test_path, name, b_test, i*10)


