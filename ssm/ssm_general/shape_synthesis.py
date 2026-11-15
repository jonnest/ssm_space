################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import numpy as np
import scipy as scp
import random
from copulas.multivariate import GaussianMultivariate,Multivariate
import warnings 

from ssm.ssm_mv_tomtec.plot_metrics import plot_correlation_heatmap
from tools.pointclouds import *


def rand_sign():
    if random.random() < 0.5:
        return 1
    else:
        return -1


def gaussian_copula(dataset, num_samples):
    '''
    This function models a single dataset D using a Gaussian copula, 
    and generates a synthetic dataset S. It returns the new, 
    synthetic version of the dataset with is samples from a Gaussian copula.
    
    https://sdv.dev/Copulas/tutorials/04_Syntehtic_Data_for_Machine_Learning.html#Generating-synthetic-data

    - maximum likelyhood estimation (based on probability) used for sampling
    
    Parameters
    ----------
    dataset : (N,M) numpy nd array
        Sets of randomly distributed parameters to couple (with Gaussian copula).
        N: number of sets (e.g. for each mode of the SSM).
        M: number of parameters (e.g. eigenvalues). 
    num_samples : integer
        Number of samples selected from the copula distribution.
        
    Returns
    -------
    synthetic : (N,M) numpy nd array
        Sets of parameters sampled from a Gaussian copula.
        N: number of sets (e.g. for each mode of the SSM).
        M: number of parameters (e.g. eigenvalues). 

    ''' 
    warnings.filterwarnings('ignore')
    
    # does find the best fitting univarite distribtution to each column
    # does not oppose the normal distribution to each column
    model = GaussianMultivariate()
    
    model.fit(dataset)

    # copula_params = model.to_dict()
    # copula_params.keys()
    # print("\nCopula params")
    # print(copula_params["univariates"])
    # print(copula_params["covariance"])
    # print(copula_params["columns"])
    # print(copula_params["type"])
    
    synthetic = model.sample(num_rows=num_samples)

    return synthetic


def shape_reconstruct(S_mean, S_ev, S_modes, nr_shapes, sigma, var_threshold, 
                      plot_path, ssm_path, properties_through_shape,
                      properties_through_parameter, S_demo, nr_scalar, use_demos,
                      include_gender,
                      shape_parameter="mean", seed_value=300,
                      S_real=None, nr_shape_param=None, name=None):
    '''
    Plot multiple shape reconstructions with normal distribution of sigma * eigenvalues.
    Only use number of modes which accumulate to the given variance threshold
        e.g. 0.95 - for 95% of the shape variation

    Parameters
    ----------        
    S_mean : (N*dim, ) numpy array
        Mean shape matrix with coordinates of mean shape.
    S_ev : (N, ) numpy array
        Eigenvalues corresponding to S_modes.
    S_modes : (N*dim, M) numpy nd array
        Mode matrix / shape variations / eigenvectors of S.
        M number of different shapes.
    nr_shapes : integer
        approximate number of shapes to plot.
    sigma : float / integer
        standard deviation of gauss normal distribution.
    var_threshold : float
        variance threshold representing the percentage of mode variance to use
        e.g. e.g. 0.95 - for 95% of the shape variation.
    plot_path : TYPE
        DESCRIPTION.
    len_shape : integer, optional
        If the shape matrix S has more than one state geometry,
        define the number of vertices to plot the shapes.
        The default is None.
    shape_parameter : string, optional
        Type of reconstruction method regarding the combination of eigenvalues. 
        The default is "mean".
    seed_value : integer, optional
        Seed of random number generator. The default is 300.
    S_real : (N*dim, nr_shapes) numpy nd array, optional
        Shape matrix for the computation of the shape parameters. The default is None.
    nr_shape_param : TYPE, optional
        DESCRIPTION. The default is None.
    name : TYPE, optional
        DESCRIPTION. The default is None.
        
    Returns
    -------
    shapes : (N*dim, n_plots) nd numpy array
        Contains the new reconstructed shape coordinates.

    '''
    # set random seed
    np.random.seed(seed_value)

    nr_modes = 0
    for i in range(len(S_ev)):
        var_sum = np.sum(S_ev[:i+1])/np.sum(S_ev)
        if var_sum < var_threshold:
            nr_modes += 1
    # add the last mode which did not enter the while loop
    nr_modes += 1
    print("Number of used modes for synthetic reconstruction :   ", nr_modes )

    
    # complete number of coordinates
    nr_coord = len(S_modes)
    # length or S_mean
    nr_v = len(S_mean)

    # intialize the shape reconstruction matrix
    # initialize metrics matrix
    S = np.empty((nr_coord, nr_shapes))
    shape = np.empty((nr_coord, nr_modes))
    
    # copula shape parameter preparation for all shapes and eigenvalues
    b_all = np.empty((nr_shapes, nr_modes))
    
    # random seed : set when always same initalization of random numbers is desired
    k = 0
    for i in range(nr_shapes):
            # iteration over used modes
            for l in range(nr_modes):
                # select method of reconstruction
                if shape_parameter == "mean":
                    # normal distribution around mean value with stdev sigma
                    b = np.random.normal(loc=0, scale=S_ev[l]*sigma)
                    # shape excitation
                    shape[:, l] = b * S_modes[:nr_coord, l]
                
                # normal distribution around positive and negative eigenvalue
                elif shape_parameter == "eigenvalues":
                    # normal distribution around mean value with stdev sigma
                    b = np.random.normal(loc=S_ev[l], scale=sigma)
                    # shape excitation with random sign
                    shape[:, l] = b * S_modes[:nr_coord, l] * rand_sign()
                
                '''
                # copula remapping of randomly distributed eigenvalues
                elif shape_parameter == "copula":
                    # shape parameter array for all modes
                    # write to general shape parameter matrix
                    b_all[i, l] = np.random.normal(loc=S_ev[l], scale=sigma)
                 '''  
            
            if shape_parameter == "mean" or shape_parameter == "eigenvalues":    
                # standard synthetic generation
                shape_var = S_mean[:nr_coord] + np.sum(shape, axis=1)
                # add synthetic mode to shape matrix
                S[:,i] = shape_var

    
    # copula synthetic generation of shape parameter
    k = 0
    if shape_parameter == "copula":   
        # variables required: S_real, nr_shape_param, name
        
        # generation of shape parameter array with Gaussian copula
        # get shape parameters of real cohort
        S_dev = S_real - np.expand_dims(S_mean, axis=1)
        b_real = get_shape_parameters(S_dev, S_modes)
        
        # add demographic values if toggled on
        if properties_through_parameter == True:
            b_real = np.hstack((b_real, S_demo))
        
        # couple and sample shape parameters on gaussian copula to secure correlations
        b_copula = np.array( gaussian_copula(b_real, nr_shapes) )
        
        # determine gender float to be male or female
        # from continuous distribution to discrete [0,1]
        if include_gender == True:
            # gender is first extra column in b_copula
            # set 0 for i<0.5 and 1 for i>0.5
            b_copula[:,-nr_scalar] = [0 if  i<0.5 else 1 for i in b_copula[:,-nr_scalar]]
        
        # plot coupled shape parameters of real cohort
        plot_real_data_shape_parameters(b_real, b_copula, nr_shape_param, plot_path, f"{name}")
        
        # plot shape parameter distribution of real and copula 
        if properties_through_parameter == True:
            demo_captions = list(use_demos.keys())
            plot_real_data_shape_parameters(b_real[:,-nr_scalar:], b_copula[:,-nr_scalar:], 
                                            nr_scalar, plot_path, f"scalar_{name}", demo_captions)
            
        # plot heat map of shape parameters
        plot_correlation_heatmap(b_real[:,:10], np.arange(len(b_real[0,:10])), plot_path,  f"{name}", "shape_param_real")
        plot_correlation_heatmap(b_copula[:,:10], np.arange(len(b_copula[0,:10])), plot_path,  f"{name}", "shape_param_copula")
        
        # loop over all synthetic shapes 
        for i in range(nr_shapes):
            # iteration over used modes
            for l in range(nr_modes):
                # shape excitation
                shape[:, l] = b_copula[i,l] * S_modes[:nr_coord, l]
                
            # standard synthetic generation
            # shape_var = np.sum(shape, axis=1)
            shape_var = S_mean[:nr_coord] + np.sum(shape, axis=1)
            
            # add synthetic mode to shape matrix
            S[:,i] = shape_var
            
        # header of shape matrix b including the demographic scalars
        # b_head = []
        # for i in range(1, len(b_real[0,:])+1):
        #     b_head.append(i)
        
        # write all shape parameters in txt file
        b_real_file = os.path.join(ssm_path, f"b_real_{name}.txt")
        np.savetxt(b_real_file, b_real)
        
        # df_b_real = pd.DataFrame(metrics, columns = metrics_caps)
        # df_b_real.to_csv(os.path.join(ssm_path, f"b_real_{name}.csv"))
        
        b_copula_file = os.path.join(ssm_path, f"b_copula_{name}.txt")
        np.savetxt(b_copula_file, b_copula)
            
    return S



def get_shape_parameters(S, S_modes):
    '''
    Get shape parameters of SSM
    
    Shape parameters of real cohort for all modes and all shapes
    Vertical: number of shapes
    Horizontal: number of modes

    Parameters
    ----------
    S : TYPE
        DESCRIPTION.
    S_modes : TYPE
        DESCRIPTION.

    Returns
    -------
    b_real : TYPE
        DESCRIPTION.

    '''
    b_real = np.empty(( len(S[0,:]), len(S_modes[0,:]) )) 
                      
    # loop over all shapes in shape matrix S
    for i in range(len(S[0,:])):
        # least squares solution of linear matrix system
        b_i, residual, _, _ = np.linalg.lstsq(S_modes, S[:,i])
        
        b_real[i,:] = b_i
        
    return b_real


def plot_real_data_shape_parameters(b_real, b_copula, nr_shape_param, plot_path, name, captions=None):
    
    # plot settings
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

    for i in range(nr_shape_param):
        ax = plt.subplot(2, int(np.ceil(nr_shape_param/2)), i + 1) 
        
        ## shapiro test for normal distribution
        # _, p_real = scp.stats.shapiro(b_real[:,i])
        # _, p_copula = scp.stats.shapiro(b_copula[:,i])
        
        ## Kolmogorov Smirnov test
        ## compare against cumulative density function of normal distribution
        _, p_real = scp.stats.kstest(b_real[:,i], scp.stats.norm.cdf)
        _, p_copula = scp.stats.kstest(b_copula[:,i], scp.stats.norm.cdf)
        
        txt1 = "p_real=" + str(round(p_real,2))
        txt2 = "p_cop=" + str(round(p_copula,2))
        
        ## Creating plot
        # plt.hist(b_real[:,i], label=txt1, **kwargs)
        # plt.hist(b_copula[:,i], label=txt2, **kwargs)
        # plt.legend()
        plt.hist(b_real[:,i], **kwargs)
        plt.hist(b_copula[:,i], **kwargs)

        if captions == None:
            ax.set_title(f"mode {i+1}")
        else: 
            ax.set_title(captions[i])
        
        # plt.ylabel("Density")
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    