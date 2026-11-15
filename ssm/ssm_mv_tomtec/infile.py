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

###############################################################################
# general settings
###############################################################################
### number of synthetic shapes
nr_shapes = 10
### workspace is name main folder
workspace = f"workspace_n{nr_shapes}"
# workspace = f"workspace_n{nr_shapes}_org87_copula_multiple"
# workspace = f"workspace_n{nr_shapes}_org87_nonlinear_single"

### process states in one shape matrix or separately in different SSMs
process_together = True


### paths and excel files to select cases and tomtec data
# workpath = "C:/Users/user/Documents/10_MV_SSM/cohorts"
workpath = "C:/Users/user/Documents/10_MV_SSM"
# workpath = "/home/user/Documents/SSM_MV_TOMTEC"


case_file = "CasesOverview_pre.xlsx"
# case_file = "CasesOverview_pre_87.xlsx"

demo_file = "Demographic_Data_anonymized.xlsx"
tomtec_file = "TomTec-Results.csv"

###############################################################################
### metrics to evaluate, no==0, yes==1
# circumferenece of annulus
circ_annulus_flag = 1
# projected area of annulus
A_annulus_flag = 1
# height of annulus in z-direction for aligned MV (rotated to xy-plane)
h_annulus_flag = 1
# ratio of annulus for aligned MV (min and max values of x/y direction)
ratio_annulus_flag = 1
# circularity index: 4*Pi*A_annulus/circ_annulus**2
ci_annulus_flag = 1
# circumference of annulus boundary
circ_orifice_flag = 1
# height of orifice in z-direction for aligned MV (rotated to xy-plane)
h_orifice_flag = 1
# ratio of orifice for aligned MV (min and max values of x/y direction)
ratio_orifice_flag = 1
# circumference of orifice boundary
ci_orifice_flag = 1
# length of anterior leaflet (midspan from annulus to orifice for y=0, for aligned MV)
len_lf_ant_flag = 1
# length of anterior leaflet (midspan from annulus to orifice for y=0, for aligned MV)
len_lf_post_flag = 1
# selected vertices for length computation : angle to normal vector of annulus plane
ang_lf_ant_flag = 1
ang_lf_post_flag = 1
# 3d anterior leaflet area
A_lf_ant_flag = 1
# 3d posterior leaflet area
A_lf_post_flag = 1
# approximated (overpredicted) 3d orifice area
A_3d_orifice_approx_flag = 1
# exact 3d orifice area; note: can cause errors due to complicated/non-robust algorithm !
A_3d_orifice_flag = 0
# cardiac measure : annulus height to width 
height_to_width_flag = 1

# not useful
# projected orifice area to orifice plane (area computation does not work correctly)
A_orifice_orif_flag = 0
# projected orifice area to annulus plane (area computation does not work correctly)
A_orifice_annu_flag = 0

###############################################################################
# routines/functions to execute
###############################################################################
### states to analyze
### possible to add multiple states e.g  ["middia", "midsys", "xxx1", "xxx2", ...]
# states = ["middia"]
states = ["middia", "midsys"]

### select specific cases of data base according to demographics
only_male = False
only_female = False

### choose between including properties through shape matrix or parameter 
### like demographic scalar features e.g. sex, weight
# process extra information
properties_through_shape = False
# only when copula method is used
properties_through_parameter = True

### choose scalar properties to include in SSM model
### make sure that all colns are filled with values
# gender with : 0 == male, 1 == female
include_gender = True
include_gender_name = "gender"
# age information
include_age = True
include_age_name = "age"
# patient height
include_height = True
include_height_name = "height"
# patient weight
include_weight = True
include_weight_name = "weight"
# pre operation new york health association (nyha)
include_pre_op_nyha = False
include_pre_op_nyha_name = "pre_op_nyha"
# body surface area
include_bsa = True
include_bsa_name = "bsa"
# body mass index
include_bmi = True
include_bmi_name = "bmi"

### plot input data
plot_original = False
plot_rows = 8
plot_colns = 13

# evaluate selected metrics of real data
get_specific_metrics_from_structure = True

# align real data for SSM analysis
ssm_alignment = False

# conduct SSM analysis on selected real data
# required to be True for shape generation
create_ssm = True

# generate synthetic shapes from SSM of real data
generate_synthetic_shapes = True

# evaluate selected metrics of synthetic data
get_synthetic_metrics = False

# evaluate and compare statistics of real and synthetic metrics
evaluate_specific_metrics = False

# plot boxplot of real and synthetic metrics
plot_boxplot = True

# plot histogram of real and synthetic metrics
plot_histogram = False

# plot basic correlation matrix; note: takes time to plot
plot_correlation = False

# plot heatmap of correlation coefficient between datasets, e.g. "pearson", "spearman", "kendall"
plot_heatmap = False
corr_method = "pearson"

# plot synthetic shapes, number of synthetic shapes to visualize
plot_synthetic = False
nr_plot_synthetic = 40

# filter synthetic data
filter_data = False



### evluate diastolic to systolic
### NOT GENERALIZED FOR MULTIPLE STATES besides diastolic and systolic
### TO USE : activate "create_ssm" and "generate_synthetic_shapes"
compare_dia_sys = True
# Procrustes disparity between systolic and diastolic annulus curve, d=0 equals same shape
procrustes_disparity = True
# compute hausdorff distance between systolic and diastolic annulus curve; time consuming !
hausdorff_distance = True
# dynamic time warpings - measure for similarity between systolic and diastolic annulus curve
dynamic_time_warping = True

###############################################################################
# SSM construction
###############################################################################
# choose alignment method : "landmark", "icp"
# icp: code exists but does not work on the current folder structure yet !
alignment = "landmark"

# variance threshold in percentage to print in console as info
print_variance = 0.6

# number of shape parameters of real data to analyze
nr_shape_param = 6

###############################################################################
# synthethic shape generation
###############################################################################
### synthetic generation of new geometries : "mean", "eigenvalues", "copula"
# only relevant for linear SSM
shape_parameter = "copula"

# use linear SSM method, nonlinear method (using Morphomatics) only works on Linux
linear_SSM_method = True

### standard deviation for standard normal distribution 
### used for "mean" and "eigenvalues" method
### in mean : scaling of the eigenvalue_i 
### in eigenvalues : sigma=stdev around the eigenvalue as the mean of the distribution
### default value : sigma=1
sigma = 1

# variance threshold for usage of modes, e.g. 1, 0.95, 0.9, 0.8, ...
# default value : var_threshold = 1.0
var_threshold = 1.0

# seed for random generator, e.g. 526562161 
# for random generated seed, do not set integer, e.g. False
seed_value = 526562161

# synthetic file names: number of digits to fill up with leading zeros
nr_digits = 5

###############################################################################
# filtering synthetic data
###############################################################################
### threshold of self intersecting faces 
### Default is 0, to capture every self-intersecting face 
remove_self_inter = False
limit_self = 0

# remove artefact in anterior leaflet
remove_artefact = False
# angle threshold between two faces next two each other
ang_threshold = 45
# vertex id of starting point
start = 1450
# width of search window
width = 20
# number of rows of search window
rows = 10

# filter according the real range of the clinical data, yes==True, no==False
filter_quartile = False

### stretch of interquartile range
### bound = q1/3 +- (1.5 + iqr_stretch) * iqr
### iqr_stretch = 0 (normal range), = -1,5 (range only from q1 to q3), ...
iqr_stretch = 0

### choose for which metrics the synthetic shapes are evaluated
### IMPORTANT : if metric was not computed beforehand, the filter does not work !
circ_annulus_filter = 1
A_annulus_filter = 1
h_annulus_filter = 1
ratio_annulus_filter = 1
ci_annulus_filter = 1
circ_orifice_filter = 1
h_orifice_filter = 1
ratio_orifice_filter = 1
ci_orifice_filter = 0
len_lf_ant_filter = 1
len_lf_post_filter = 1
ang_lf_ant_filter = 1
ang_lf_post_filter = 1
A_lf_ant_filter = 1
A_lf_post_filter = 1
A_3d_orifice_filter = 1

A_orifice_orif_filter = 0
A_orifice_annu_filter = 0

###############################################################################
# do not change the following parameters for the SSM Mitral Valve project
# based on TomTec segmented data
###############################################################################
# mitral valve geometry input parameters
###############################################################################
# vertex id's of commissure points
lat_com = 1580
med_com = 1540
# vertex id for horizontal axis from center of annulus, to build coordinate system
v_hori = 0

# number of vertices over circumference
nr_v_circ = 80
# number of vertice layers
nr_layers = 20

# boundary id's
bnd_id_annulus = 0
bnd_id_orifice = 1

### vertex id's on boundaries of leaflets
### specifically for MV's segmentations from TomTec
id_annu_ant = np.array(np.linspace(20,59,40), dtype=int)
id_orif_ant = np.array(np.linspace(1540,1579,40), dtype=int)    
id_annu_post = np.array(np.append(np.linspace(0,19,20), np.linspace(60,79,20)), dtype=int)
id_orif_post = np.array(np.append(np.linspace(1520,1539,20), np.linspace(1580,1599,20)), dtype=int)
id_orif = np.array(np.linspace(1520,1599,80), dtype=int)  

# EOF