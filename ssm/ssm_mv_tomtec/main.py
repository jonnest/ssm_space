################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Python routine for evaluating metrics of real MV and synthetic data
Generating SSM based on real data to create synthetic geometrics
Comparison of metric evaluation with TomTec results for metrics as validation


'''
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import sys

from read_metrics import * 
from plot_metrics import * 
from synthetic_cohort import create_synthetic_cohort, plot_synthetic_sample
from tools.io_utils import filtering_func, get_number_of_items, readTriangularSurfaceMesh
from get_metrics import get_real_data_metrics, get_synthetic_data_metrics
from ssm_setup import align_real_data, read_aligned_data, ssm_setup, \
                        plot_input_data, get_mesh_list

from filter_data import remove_filtered_synthetic_data, filter_metrics, \
                        find_outliers, get_hausdorff_means, get_dynamic_time_warpings, \
                        get_procrustes_disparity

from infile import * 

try:
    from ssm.ssm_general.ssm_nonlinear import nonlinear_synthesis
except:
    print("\n!!! Morphomatics library can not be included !!!\nCheck installations.\n")

###############################################################################
# warnings : plausibility checks
###############################################################################

if (generate_synthetic_shapes == True) and (shape_parameter == "copula"):
    
    if nr_shapes <= nr_shape_param:
        
        print("\n!!! Warning !!!\n\nnr_shape_param must be smaller than nr_shapes !\n")
        print("*** End of program ***\n")
        sys.exit()

# if evaluate_specific_metrics == False and ( plot_histogram or plot_spearman):
    
#     print("\n!!! Warning !!!\n\nTo plot statistics 'evaluate_specific_metrics' has to be set 'True' !\n")
#     print("*** End of program ***\n")
#     sys.exit()
        
if only_male == True and only_female == True:
    print("\n!!! Warning !!!\n\nNot possible to have 'only_male' and 'only_female' both on 'True' !\n")
    print("*** End of program ***\n")
    sys.exit()
    
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
# folder to save infile properties
infile_name = "infile"

# paths resulting from workpath with respective folders
project_path = os.path.abspath(workpath + "/" + workspace + "/")
# subdirectoreis of project path
data_path = os.path.abspath(project_path +  "/" + case_name + "/")
metric_path = os.path.abspath(project_path +  "/" + save_name + "/")
ssm_path = os.path.abspath(project_path +  "/" + ssm_name + "/")
syn_path = os.path.abspath(project_path +  "/" + syn_name + "/")
plot_path = os.path.abspath(project_path +  "/" + plot_name + "/")
stats_path = os.path.abspath(project_path +  "/" + stats_name + "/")
infile_path = os.path.abspath(project_path +  "/" + infile_name + "/")
# demographics 
demo_path = os.path.abspath(project_path +  "/" + demo_file + "/")
# tomtec evalutation file
tomtec_path = os.path.abspath(project_path + "/" + tomtec_file)
# path to selection file of cases
select_path = os.path.abspath(project_path +  "/" + case_file + "/")

# select real cases to use from case_file
cases = pd.read_excel(select_path)
use_cases = cases.loc[cases["use"] == 1]
use_cases = use_cases["case"].to_numpy()
# number of cases
nr_cases = len(cases[cases["use"]==1])

# read demographic file
df_demo = pd.read_excel(demo_path)
df_demo_select = df_demo.loc[df_demo["record_id"].isin(use_cases)]

# overwrite the already selected use_cases 
# adapted on demographics, e.g. only male/female
if only_male == True:
    df_demo_select = df_demo_select.loc[df_demo_select["gender"] == 0]
    use_cases = np.array(df_demo_select["record_id"])
    nr_cases = len(use_cases)
    print("\nUse :    ", len(use_cases), " cases of real data")
    
elif only_female == True:
    df_demo_select = df_demo_select.loc[df_demo_select["gender"] == 1]
    use_cases = np.array(df_demo_select["record_id"])
    nr_cases = len(use_cases)
    print("\nUse :    ", len(use_cases), " cases of real data")

elif only_male == False and only_female == False: 
    print("\nUse :    ", len(use_cases), " cases of real data")
    
    # get male/female case numbers
    df_males = df_demo_select.loc[df_demo_select["gender"] == 0]
    df_females = df_demo_select.loc[df_demo_select["gender"] == 1]
    male_cases = np.array(df_males["record_id"])
    female_cases = np.array(df_females["record_id"])
    
    # get binary arrays for True if case is male/female
    male_bool_org = [True if num in male_cases else False for num in use_cases]
    female_bool_org = [True if num in female_cases else False for num in use_cases]

print("\nCreate : ", nr_shapes, " synthetic cases")

###############################################################################
# warnings : plausibility checks
##############################################################################################################################################################
if nr_cases <= 3 and shape_parameter == "copula":
    print("\n!!! Warning !!!\n\nnr_cases must be at least equal to 4 using the 'copula' method!\n")
    print("*** End of program ***\n")
    sys.exit()
elif nr_cases <= 2 and shape_parameter == "mean":
    print("\n!!! Warning !!!\n\nnr_cases must be at least equal to 3 when using the 'mean' method !\n")
    print("*** End of program ***\n")
    sys.exit()
###############################################################################
# dictionary initialization
###############################################################################
# dictionary of metrics selection from infile.py
metric_names = {
    "circ_annulus" : circ_annulus_flag,
    "A_annulus" : A_annulus_flag,
    "h_annulus" : h_annulus_flag,
    "ratio_annulus" : ratio_annulus_flag,
    "ci_annulus" : ci_annulus_flag,
    "circ_orifice" : circ_orifice_flag,
    "A_orifice_orif" : A_orifice_orif_flag,
    "A_orifice_annu" : A_orifice_annu_flag,
    "h_orifice" : h_orifice_flag,
    "ratio_orifice" : ratio_orifice_flag,
    "ci_orifice" : ci_orifice_flag,
    "len_lf_ant" : len_lf_ant_flag,
    "len_lf_post" : len_lf_post_flag,
    "ang_lf_ant" : ang_lf_ant_flag,
    "ang_lf_post" : ang_lf_post_flag,
    "A_lf_ant" : A_lf_ant_flag,
    "A_lf_post" : A_lf_post_flag,
    "A_3d_orifice_approx" : A_3d_orifice_approx_flag,
    "A_3d_orifice" : A_3d_orifice_flag,
    "height_to_width" : height_to_width_flag
}

# corresponding units to upper dictionary of metrics
metric_units = {
    "circ_annulus" : "[mm]",
    "A_annulus" : "[mm^2]",
    "h_annulus" : "[mm]",
    "ratio_annulus" : "[-]",
    "ci_annulus" : "[-]",
    "circ_orifice" : "[mm]",
    "A_orifice_orif" : "[mm^2]",
    "A_orifice_annu" : "[mm^2]",
    "h_orifice" : "[mm]",
    "ratio_orifice" : "[-]",
    "ci_orifice" : "[-]",
    "len_lf_ant" : "[mm]",
    "len_lf_post" : "[mm]",
    "ang_lf_ant" : "[°]",
    "ang_lf_post" : "[°]",
    "A_lf_ant" : "[mm^2]",
    "A_lf_post" : "[mm^2]",
    "A_3d_orifice_approx" : "[mm^2]",
    "A_3d_orifice" : "[mm^2]",
    "height_to_width" : "[-]"
}

# dictionary of metrics to filter
metrices_to_filter_all = {
    "circ_annulus" : circ_annulus_filter,
    "A_annulus" : A_annulus_filter,
    "h_annulus" : h_annulus_filter,
    "ratio_annulus" : ratio_annulus_filter,
    "ci_annulus" : ci_annulus_filter,
    "circ_orifice" : circ_orifice_filter,
    "A_orifice_orif" : A_orifice_orif_filter,
    "A_orifice_annu" : A_orifice_annu_filter,
    "h_orifice" : h_orifice_filter,
    "ratio_orifice" : ratio_orifice_filter,
    "ci_orifice" : ci_orifice_filter,
    "len_lf_ant" : len_lf_ant_filter,
    "len_lf_post" : len_lf_post_filter,
    "ang_lf_ant" : ang_lf_ant_filter,
    "ang_lf_post" : ang_lf_post_filter,
    "A_lf_ant" : A_lf_ant_filter,
    "A_lf_post" : A_lf_post_filter,
    "A_3d_orifice" : A_3d_orifice_filter,
}

# allocation of mitral valve geometry input parameters in dictionary
parameters = {
    "lat_com" : lat_com,
    "med_com" : med_com,
    "v_hori" : v_hori,
    "nr_v_circ" : nr_v_circ,
    "nr_layers" : nr_layers,
    "bnd_id_annulus" : bnd_id_annulus,
    "bnd_id_orifice" : bnd_id_orifice,
    "id_annu_ant" : id_annu_ant,
    "id_orif_ant" : id_orif_ant,    
    "id_annu_post" : id_annu_post,
    "id_orif_post" : id_orif_post,
    "id_orif" : id_orif
}

### demographic scalar features
### index end of row indicates the column in the b_copula, but only if all are set on 'True'
### do not change order / at least gender_name needs to be first in S_demo, demographics is read via loop over keys
demographics = {
    include_gender_name : include_gender,           # -7
    include_age_name : include_age,                 # -6
    include_height_name : include_height,           # -5
    include_weight_name : include_weight,           # -4
    include_pre_op_nyha_name : include_pre_op_nyha, # -3
    include_bsa_name : include_bsa,                 # -2
    include_bmi_name : include_bmi                  # -1
}

###############################################################################
# setup of evaluation
###############################################################################
# filter metric names to use_metrics with all properties to evaluate
use_metrics = dict(filter(filtering_func, metric_names.items()))

# get number of properties to analyze from metric_names
nr_props = get_number_of_items(metric_names)

# get number of items to filter from metrices_to_filter
metrices_to_filter = dict(filter(filtering_func, metrices_to_filter_all.items()))
nr_filt = get_number_of_items(metrices_to_filter)
print("\nNumber of items to filter : ", nr_filt)

print("\nWorkspace for analysis :   ", project_path)
t_start = time.time()

# get metric captions
metrics_caps = []
for count, key in enumerate(use_metrics.keys()):
    metrics_caps.append(key)

# counter for units
print("\n- Metrics to use:")
x = np.linspace(1, nr_props, nr_props)
use_units = []
for key in use_metrics.keys():
    if key in metric_units.keys():
        use_units.append(metric_units[key])
        print(key)
        
# check if plot folder exists
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# check if stats folder exists
if not os.path.exists(stats_path):
    os.makedirs(stats_path)
    

# count number of scalar values in ssm
use_demos = dict(filter(filtering_func, demographics.items()))
nr_scalar = get_number_of_items(use_demos)
print("\nNumber of scalars values to include in the SSM : ", nr_scalar)

# read in scalar properties to S_demo
S_demo = np.empty((nr_cases, nr_scalar))
for s, key in enumerate(use_demos.keys()):
    S_demo[:,s] = df_demo_select[key].to_numpy()

###############################################################################
# functions to execute in the workflow
###############################################################################
# process all states in one single SSM framework together
if process_together == True:
    
    ### align original shapes for SSM
    if ssm_alignment == True:
        # align real data for SSM
        for state in states:
            align_real_data(data_path, use_cases, parameters, alignment, name=state)
        
        
    if create_ssm == True:
        # init files for each geometry
        init_mesh_files = []
        S_all = []
        for j, state in enumerate(states):
            # save mesh files for initialization in list
            init_mesh_files.append(os.path.abspath(data_path + "/" + str(use_cases[0]) + f"/{state}/MV_{state}.obj"))
        
            # create shape matrix for all real diastolic and systolic cases
            S_i = read_aligned_data(nr_cases, data_path, use_cases, init_mesh_files[j], name=state)           
            
            # save shape matrix of one state in list
            S_all.append(S_i)    

        # stack all states in shape matrix
        S = np.vstack(S_all)
        
        # save in matrix for later evaluation
        org_shapes = S
            
        # construct SSM and save parameters
        ssm_setup(S, ssm_path, print_variance, name="all")
            
        
    # generate the synthetic shapes via a combined, multiple state shape matrix S
    if generate_synthetic_shapes == True:                  
        # create synthetic shapes and save shape matrices of synthetic data
        # indicate number of states processed in init_mesh_files
        syn_shapes, _ = create_synthetic_cohort(ssm_path, syn_path, plot_path, init_mesh_files,
                                                nr_shapes, sigma, var_threshold, seed_value,
                                                shape_parameter, S, nr_digits, nr_shape_param, nr_scalar, 
                                                use_demos, include_gender,
                                                properties_through_shape, 
                                                properties_through_parameter, S_demo, "all", states)


###############################################################################
# save synthetic shapes of states in list for evaluation
if process_together == False:
    syn_shapes = []
    org_shapes = []

### loop over states
for state in states:   
    ### initialization file for metric matrices
    init_mesh_file = os.path.abspath(data_path + "/" + str(use_cases[0]) + f"/{state}/MV_{state}.obj")
    init_mesh_files = [init_mesh_file]

    ### get metrics of original data
    if get_specific_metrics_from_structure == True:
        # get metrics of real data from directory structure
        get_real_data_metrics(nr_cases, nr_props, use_metrics, metrics_caps, use_cases,
                              data_path, project_path, parameters, metric_path, state, 
                              remove_artefact,
                              start, width, rows, nr_v_circ, ang_threshold)


    ### align original shapes for SSM
    if ssm_alignment == True:
        # align real data for SSM
        align_real_data(data_path, use_cases, parameters, alignment, name=state)


    ### process every state in a separate SSM framework
    if process_together == False:
        if create_ssm == True:
            # create shape matrix for all real diastolic and systolic cases
            S = read_aligned_data(nr_cases, data_path, use_cases, init_mesh_file, name=state)           
                    
            # save originial shapes to list
            org_shapes.append(S)
            
            # process scalar properties in shape matrix S
            if properties_through_shape == True:
                # stack scalar properties from S_demo to shape matrix S
                S = np.vstack((S, S_demo))
                
            # construct SSM and save parameters
            ssm_setup(S, ssm_path, print_variance, name=state)
            
            # plot original input data, dependent if scalars are stacked in shape matrix
            if plot_original == True and properties_through_shape == False:
                # include complete shape matrix
                plot_input_data(S, init_mesh_file, use_cases, plot_rows, plot_colns)
            elif plot_original == True and properties_through_shape == True:
                # exclude scalar values from shape matrix
                plot_input_data(S[:-nr_scalar,:], init_mesh_file, use_cases, plot_rows, plot_colns)
                
                
        if generate_synthetic_shapes == True:      
            # linear shape synthesis
            if linear_SSM_method == True:
                # create synthetic shapes and save shape matrices of synthetic data
                syn_shapes_state, _ = create_synthetic_cohort(ssm_path, syn_path, plot_path, init_mesh_files,
                                                        nr_shapes, sigma, var_threshold, seed_value,
                                                        shape_parameter, S, nr_digits, nr_shape_param, nr_scalar, 
                                                        use_demos, include_gender,
                                                        properties_through_shape, 
                                                        properties_through_parameter, S_demo, state)
            
            # non linear SSM and shape synthesis
            else:
                # get list of paths for original meshes to include in the non-linear SSM
                mesh_list = get_mesh_list(data_path, use_cases, state)
                # get synthetic shapes based on the non-linear SSM and randomized shape reconstruction (mean, gauss)
                syn_shapes_state = nonlinear_synthesis(mesh_list, syn_path, nr_shapes, sigma, var_threshold,
                                                        seed_value, nr_digits, print_variance, parameters, state)

            # save synthetic shapes to list
            syn_shapes.append(syn_shapes_state)
            
            # plot shapes after generation of shapes
            if plot_synthetic == True:
                # plot a sample of the new generated synthetic data
                plot_synthetic_sample(syn_shapes_state, init_mesh_file, nr_plot_synthetic)


    ### get metrics of synthetic shapes
    if get_synthetic_metrics == True:
        # get metrics of synthetic data
        ext = ""
        get_synthetic_data_metrics(nr_shapes, nr_props, use_metrics, metrics_caps, syn_path,
                                   data_path, project_path, parameters, metric_path, 
                                   nr_scalar, state, ext, remove_artefact,
                                   start, width, rows, nr_v_circ, ang_threshold)


    ### evaluation and analysis of metrics
    if evaluate_specific_metrics == True:
        # read metrics of real data
        m_org = read_real_metrics(metric_path, name=state)
        # read metrics of synthetic data
        m_syn = read_synthetic_metrics(metric_path, name=state)
        # read metrics of tomtec results file
        m_tom = read_tomtec_evaluation(tomtec_path, nr_props, name=state)
        
        # boxplot
        if plot_boxplot == True:
            boxplot_metrics(m_org, m_syn, m_tom, use_metrics, use_units, metrics_caps,
                                nr_props, plot_path, name=state)

        # histogram
        if plot_histogram == True:
            histogram_metrics(m_org, m_syn, m_tom, use_metrics, use_units, metrics_caps,
                                  nr_props, plot_path, stats_path, name=state)

        # basic correlation matrix
        if plot_correlation == True:
            # correlation plots of the metrics colored by sex
            if only_male == False and only_female == False and shape_parameter == "copula":
                # colored correlation plot by sex
                key = "sex"
                names = ["male", "female"]
                state_org_list = [m_org[male_bool_org], m_org[female_bool_org]]
                # sex in synthetic data - get from b_copula
                if process_together == False:
                    b_copula = np.loadtxt(os.path.join(ssm_path, f"b_copula_{state}.txt"))
                else:
                    b_copula = np.loadtxt(os.path.join(ssm_path, f"b_copula_all.txt"))
                
                # assign column of sex to array
                sex_syn = b_copula[:,-nr_scalar]
                male_bool_syn = [True if num == 0 else False for num in sex_syn]
                female_bool_syn = [True if num == 1 else False for num in sex_syn]
                state_syn_list = [m_syn[male_bool_syn], m_syn[female_bool_syn]]
                
                # correlation plot with colored sex
                plot_correlation_colored_by_key(state_org_list, metrics_caps, plot_path, key, names, f"{state}_org")
                plot_correlation_colored_by_key(state_syn_list, metrics_caps, plot_path, key, names, f"{state}_syn")
            
            # standard correlation plots of the metrics
            else:
                # plot correlation when only male or only female analysis
                plot_correlation_matrix(m_org, metrics_caps, plot_path, name=state, data="org")
                plot_correlation_matrix(m_syn, metrics_caps, plot_path, name=state, data="syn")

        # correlation heatmap
        if plot_heatmap == True:
            r_org = plot_correlation_heatmap(m_org, metrics_caps, plot_path, name=state, data="org", corr_method=corr_method)
            r_syn = plot_correlation_heatmap(m_syn, metrics_caps, plot_path, name=state, data="syn", corr_method=corr_method)
            
            print(f"\n\nStatistical evaluation for {state} - comparison of Spearman Correlation\n")
            print(f"mean(r_org)      ", np.mean(np.mean(r_org)))
            print(f"mean(r_syn)      ", np.mean(np.mean(r_syn)))
            print(f"difference       ", round(np.mean(np.mean(r_org))-np.mean(np.mean(r_syn)),3 ))
            print(f"difference in %  ", round((np.mean(np.mean(r_org))-np.mean(np.mean(r_syn))/np.mean(np.mean(r_org))),3 ))
            
            correlation_file = os.path.abspath(stats_path +  "/" + f"comparison_spearman_correl_{state}.txt")
            with open(correlation_file, 'w') as f:
                f.writelines(f"r_org-r_mean {round(np.mean(np.mean(r_org))-np.mean(np.mean(r_syn)),3 )}\n")
                f.writelines(f"rel {round((np.mean(np.mean(r_org))-np.mean(np.mean(r_syn))/np.mean(np.mean(r_org))),3 )}\n")
       
       # correlation between demographics and metrics (selected) 
       
        
    print("\n\nFilter data\n\n")
    ### filter synthetic data
    if filter_data == True:
        # read metrics of real data
        m_org = read_real_metrics(metric_path, name=state)
        # read metrics of synthetic data
        m_syn = read_synthetic_metrics(metric_path, name=state)
        # read metrics of tomtec results file
        m_tom = read_tomtec_evaluation(tomtec_path, nr_props, name=state)
        
        # move synthetic meshes with self-intersection to different folder
        if remove_self_inter == True:
            # remove selfintersecting meshes to different folder
            remove_filtered_synthetic_data(syn_path, state, "intersect", limit_self)

        # move synthetic meshes with anterior artefact to different folder
        if remove_artefact == True and state=="middia":
            # only for diastolic state
            remove_filtered_synthetic_data(syn_path, state, "artefact",
                                           start, width, rows, nr_v_circ, ang_threshold)
                        
        # find outliers
        if filter_quartile == True:
            filter_metrics(syn_path, m_org, m_syn, iqr_stretch, metrices_to_filter, name=state)   
        
        
        # analysis of filtered data
        if plot_boxplot == True:
            ext = "_filtered"
            # get metrics of synthetic data
            get_synthetic_data_metrics(nr_shapes, nr_props, use_metrics, metrics_caps, syn_path,
                                        data_path, project_path, parameters, metric_path, nr_scalar, state, ext,
                                        remove_artefact,
                                        start, width, rows, nr_v_circ, ang_threshold)
            
            # read metrics of FILTERED synthetic data
            m_syn_fil = read_synthetic_metrics(metric_path, name=state, ext=ext)

            # boxplot of filtered data
            boxplot_metrics(m_org, m_syn_fil, m_tom, use_metrics, use_units, metrics_caps,
                                nr_props, plot_path, name=state+ext)   


        # TODO : further evaluation plots for filtered data INSERT here
        # see above



###############################################################################
### evaluation state to state : diastolic to systolic
### comparison of state to state behaviour between original and synthetic data

if compare_dia_sys == True and create_ssm == True and generate_synthetic_shapes == True:
    print("\n\nCompare diastolic and systolic state !\n")
    print("States processed together in one SSM : ", process_together)
    
    ### read metrics of real data
    m_org_dia = read_real_metrics(metric_path, name="middia")
    m_org_sys = read_real_metrics(metric_path, name="midsys")
    
    ### read metrics of synthetic data
    m_syn_dia = read_synthetic_metrics(metric_path, name="middia")
    m_syn_sys = read_synthetic_metrics(metric_path, name="midsys")
    
    ### relative deviation of metrics between diastolic and systolic state
    m_org_rel = (m_org_dia - m_org_sys) / m_org_dia*100
    m_syn_rel = (m_syn_dia - m_syn_sys) / m_syn_dia*100
    
    ### boxplot of relative deviation between diastolic and systolic state
    if plot_boxplot == True:
        boxplot_metrics(m_org_rel, m_syn_rel, {}, use_metrics, ["relative deviation in %"]*nr_props, metrics_caps,
                        nr_props, plot_path, name="relative_dia_sys")
    
    ### correlation matrix of diastolic and systolic features against each other
    org_list = [m_org_dia, m_org_sys]
    syn_list = [m_syn_dia, m_syn_sys]
    ### basic correlation matrix
    if plot_correlation == True:
        key = "state"
        plot_correlation_colored_by_key(org_list, metrics_caps, plot_path, key, states, "org")
        plot_correlation_colored_by_key(syn_list, metrics_caps, plot_path, key, states, "syn")

    ### scatterplot dia to sys
    plot_states_scatter(m_org_dia, m_org_sys, m_syn_dia, m_syn_sys, metrics_caps, plot_path)
    
    
    ### get hausdorff distance of annulus curve for diastolic and systolic state
    if type(syn_shapes) == list and syn_shapes != []:
        dia_shapes_org = org_shapes[0]
        sys_shapes_org = org_shapes[1]
        
        dia_shapes_syn = syn_shapes[0]
        sys_shapes_syn = syn_shapes[1]
        
    else:
        # syn_shapes together in one shape matrix
        # initialization : NOT GENERALIZED
        v, f = readTriangularSurfaceMesh(init_mesh_files[0])
        
        dia_shapes_org = org_shapes[:len(v)*3,:]
        sys_shapes_org = org_shapes[len(v)*3:,:]
      
        dia_shapes_syn = syn_shapes[:len(v)*3,:]
        sys_shapes_syn = syn_shapes[len(v)*3:,:]

    ### initialize similarity file lines
    sim_head = []
    sim_mean = []
    sim_std = []
    
    ### get Procrustes disparity between annulus curve for diastolic and systolic state
    if procrustes_disparity == True:
        d_org = get_procrustes_disparity(dia_shapes_org, sys_shapes_org, init_mesh_files, metric_path, "org")
        d_syn = get_procrustes_disparity(dia_shapes_syn, sys_shapes_syn, init_mesh_files, metric_path, "syn")
        print("\nDisparity mean org  ", round(d_org.mean(),3))
        print("Disparity std  org  ", round(d_org.std(),3))
        print("Disparity mean syn  ", round(d_syn.mean(),3))
        print("Disparity std  syn  ", round(d_syn.std(),3))
        sim_head.append("procrustes_org")
        sim_head.append("procrustes_syn")
        sim_mean.append(d_org.mean())
        sim_std.append(d_org.std())
        sim_mean.append(d_syn.mean())
        sim_std.append(d_syn.std())
        
    ### get all Hausdorff distance means between annulus curve for diastolic and systolic state
    ### time consuming due to PyMeshLab usage, saves all shapes as files again etc.
    if hausdorff_distance == True:
        hd_means_org = get_hausdorff_means(dia_shapes_org, sys_shapes_org, init_mesh_files, metric_path, "org")
        hd_means_syn = get_hausdorff_means(dia_shapes_syn, sys_shapes_syn, init_mesh_files, metric_path, "syn")
        print("\nHd mean org  ", hd_means_org.mean())
        print("Hd std  org  ", hd_means_org.std())
        print("Hd mean syn  ", hd_means_syn.mean())
        print("Hd std  syn  ", hd_means_syn.std())
        sim_head.append("hausdorff_org")
        sim_head.append("hausdorff_syn")
        sim_mean.append(hd_means_org.mean())
        sim_std.append(hd_means_org.std())
        sim_mean.append(hd_means_syn.mean())
        sim_std.append(hd_means_syn.std())
        
    ### dynamic time warpings - measure for similarity
    if dynamic_time_warping == True:
        dtws_org = get_dynamic_time_warpings(dia_shapes_org, sys_shapes_org, init_mesh_files, metric_path, "org")
        dtws_syn = get_dynamic_time_warpings(dia_shapes_syn, sys_shapes_syn, init_mesh_files, metric_path, "syn") 
        print("\nDtw mean org  ", dtws_org.mean())
        print("Dtw std  org  ", dtws_org.std())
        print("Dtw mean syn  ", dtws_syn.mean())
        print("Dtw std  syn  ", dtws_syn.std())
        sim_head.append("dtw_org")
        sim_head.append("dtw_syn")
        sim_mean.append(dtws_org.mean())
        sim_std.append(dtws_org.std())
        sim_mean.append(dtws_syn.mean())
        sim_std.append(dtws_syn.std())
        
    ### write data to similarity metrics file
    sim_array = np.vstack((sim_mean, sim_std))
    np.savetxt(metric_path + "/similarities_mean_std.txt", sim_array, header=" ".join(sim_head))
    df_sim = pd.DataFrame(sim_array, columns = sim_head)
    df_sim.to_csv(os.path.join(metric_path, "similarities_mean_std.csv"))
        
    
    
###############################################################################
# save settings and parameters in file
###############################################################################
t_end = time.time()
# total time of computation
print("\n\nTotal time : ", round(t_end-t_start,3)," seconds")

# file name to save all parameters
if not os.path.exists(infile_path):
   os.makedirs(infile_path)
   
save_parameter_file = os.path.abspath(infile_path +  "/" + "save_parameter_file.txt")

# dictionary of settings to save
save_parameter_dict = {
    "project_path" : project_path,
    "nr_cases" : nr_cases,
    "nr_shapes" : nr_shapes,
    "states" : states,
    "process_together" : process_together,
    "nr_props" : nr_props,
    "alignment" : alignment,
    "sigma" : sigma,
    "shape_parameter" : shape_parameter,
    "var_threshold" : var_threshold,
    "only_male" : only_male,
    "only_female" : only_female,
    "properties_through_shape" : properties_through_shape,
    "properties_through_parameter" : properties_through_parameter,
    "include_age" : include_age,
    "include_gender" : include_gender,
    "include_bsa" : include_bsa,
    "include_bmi" : include_bmi,
    "seed_value" : seed_value,
    "filter_quartile" : filter_quartile,
    "iqr_stretch" : iqr_stretch,
    "total_time_in_s" : round(t_end-t_start,3),
}

# write parameter dictionary to file
with open(save_parameter_file, 'w') as f:
    f.writelines("{'\n")
    for key in save_parameter_dict:
        if isinstance(save_parameter_dict[key], str):
            f.writelines("'" + key + "' :  '" + str(save_parameter_dict[key]) + "',\n")
        else: 
            f.writelines("'" + key + "' :  " + str(save_parameter_dict[key]) + ",\n")
    f.writelines("'}")

# copy infile.py to infile folder
with open("infile.py", "r") as file1:
    with open(infile_path + "/infile.txt", "w") as file2:
        file2.write(file1.read())
        
# EOF