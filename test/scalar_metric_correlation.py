################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Infile for Statistical Shape Analysis for analyzing Mitral Valve data from
TomTec. 

Only change parameters, paths and settings in this file.

Select which subroutines in the main.py file are executed.

'''

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import seaborn as sns
import sys
import pyvista as pv
import scipy as scp

from ssm.ssm_general.shape_synthesis import *
from tools.io_utils import *
from ssm.ssm_mv_tomtec.infile import * 
from ssm.ssm_mv_tomtec.plot_metrics import * 

###############################################################################
# general settings
###############################################################################
# number of synthetic shapes
nr_shapes = 100

# workspace is name main folder
workspace = f"workspace_n{nr_shapes}_single"

# paths and excel files to select cases and tomtec data
workpath = "C:/Users/user/Documents/10_MV_SSM"
##workpath = "/home/user/Documents/mv_ssm_loc/MV_SSM/"
case_file = "CasesOverview_pre.xlsx"
tomtec_file = "TomTec-Results.csv"


###############################################################################
# Paths 
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
plot_name = "plots"
# folder to save all statistical evaluations
stats_name = "stats"

# paths resulting from workpath with respective folders
project_path = os.path.abspath(workpath + "/" + workspace + "/")
# subdirectoreis of project path
data_path = os.path.abspath(project_path +  "/" + case_name + "/")
metric_path = os.path.abspath(project_path +  "/" + save_name + "/")
ssm_path = os.path.abspath(project_path +  "/" + ssm_name + "/")
syn_path = os.path.abspath(project_path +  "/" + syn_name + "/")
plot_path = os.path.abspath(project_path +  "/" + plot_name + "/")
stats_path = os.path.abspath(project_path +  "/" + stats_name + "/")
# demographics 
demo_path = os.path.abspath(project_path +  "/" + demo_file + "/")
# tomtec evalutation file
tomtec_path = os.path.abspath(project_path + "/" + tomtec_file)
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


###############################################################################

# read shape parameters
b_org_file = os.path.join(plot_path, "b_real_middia.txt")
b_syn_file = os.path.join(plot_path, "b_copula_middia.txt")

b_org = np.loadtxt(b_org_file)
b_syn = np.loadtxt(b_syn_file)

# compare bmi
syn_bmi_calc = b_syn[:,-3]/b_syn[:,-4]**2
syn_bmi_map = b_syn[:,-1]
syn_bmi_diff = (syn_bmi_calc - syn_bmi_map) / syn_bmi_calc

print("\nsyn_bmi_diff (mean) in %   ", round(syn_bmi_diff.mean()*100,2))
print("syn_bmi_diff (std) in %    ", round(syn_bmi_diff.std()*100,2))
print("syn_bmi_diff (max) in %    ", round(max(syn_bmi_diff)*100,2))

# compare bsa
syn_bsa_calc = np.sqrt(b_syn[:,-3]*100*b_syn[:,-4]/3600)
syn_bsa_map = b_syn[:,-2]
syn_bsa_diff = (syn_bsa_calc - syn_bsa_map) / syn_bsa_calc

print("\nsyn_bsa_diff (mean) in %   ", round(syn_bsa_diff.mean()*100,2))
print("syn_bsa_diff (std) in %    ", round(syn_bsa_diff.std()*100,2))
print("syn_bsa_diff (max) in %    ", round(max(syn_bsa_diff)*100,2))

###############################################################################
# metrics
m_org_file = os.path.join(metric_path, "metrics_middia_original.txt")
m_syn_file = os.path.join(metric_path, "metrics_middia_synthetic.txt")

m_org = np.loadtxt(m_org_file)
m_syn = np.loadtxt(m_syn_file)
###############################################################################
# prepare correlation matrix
data_org = np.hstack((b_org[:,-4:], m_org[:,:2]))
data_syn = np.hstack((b_syn[:,-4:], m_syn[:,:2]))


metrics_caps = ["height", "weight", "bsa", "bmi", "circ_annulus", "A_annulus"]


###############################################################################

def plot_correlation_matrix(np_data, metrics_caps, plot_path, name, data):
    '''
    Plot the basic correlation matrix between all parameters.

    Parameters
    ----------
    np_org : (N, nr_props) numpy nd array
        Array of metrics for N data shapes and nr_props properties.
    metrics_caps : list
        List with strings of the metric captions.
    plot_path : string
        Absolute path to the plot directory.
    name : string
        Name of the data, e.g. the state (diastolic or systolic).
    data : string
        Category of data, e.g. real or synthetic ('real' or 'syn').

    Returns
    -------
    None.

    '''

    print(f"\nPlot basic correlation matrix of metrics for {name}_{data}...")
    # basic correlation between metrics of data
    df_data = pd.DataFrame()

    # import np data to dataframe
    for i in range(len(metrics_caps)):
        df_data[metrics_caps[i]] = np_data[:, i]

    # pairplot correlation basic matrix
    sns.pairplot(df_data, kind="reg")

    correl_file = os.path.abspath(plot_path + "/" + f"{name}_{data}_correlation.png")
    plt.savefig(correl_file, dpi=300)

    return



plot_correlation_matrix(data_org, metrics_caps, plot_path, name="demo_metrics", data="org")

plot_correlation_matrix(data_syn, metrics_caps, plot_path, name="demo_metrics", data="syn")
















