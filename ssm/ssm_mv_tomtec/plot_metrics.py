################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns

from tools.plots import display_correlation


def boxplot_metrics(np_org, np_syn, df_tom, use_metrics, use_units,
                    metrics_caps, nr_props, plot_path, name):
    '''
    Create boxplot of distribution of metrics for original (real), synthetic, 
    and tomtec data.

    Parameters
    ----------
    np_org : (N, nr_props) numpy nd array
        Array of metrics for N real data shapes and nr_props properties.
    np_syn : (M, nr_props) numpy nd array
        Array of metrics for M synthetic data shapes and nr_props properties.
    df_tom : pandas.Dataframe
        Dataframe with TomTec metrics for N real data shapes and P TomTec result
        properties.
    use_metrics : dictionary
        Dictionary of all metrics to use.
    use_units : list
        List with strings of the metric units.
    metrics_caps : list
        List with strings of the metric captions.
    nr_props : integer
        Number of properties / metrics to evaluate.
    plot_path : string
        Absolute path to the plot directory.
    name : string
        Name of the data, e.g. the state (diastolic or systolic).

    Returns
    -------
    None.

    '''
    print(f"\nPlot boxplot of metrics for {name}...")
    print("Length of syn ", len(np_syn))
    # plot settings
    # inner size structure of figures
    top = 0.95
    bottom = 0.055
    left = 0.05
    right = 0.98
    hspace = 0.25
    wspace = 0.8
    # fonts
    font = {'size': 8}

    # subplot for each property
    fig = plt.figure(figsize=(11, 7))
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        hspace=hspace, wspace=wspace)

    plt.rc('font', **font)

    data = []
    for i, key in enumerate(use_metrics.keys()):
        # fill with colors
        boxcolors = ['lightgreen', 'seagreen']
        x_ticks_labels = [f"real", f"syn"]

        # allocate data for boxplot
        data = [np_org[:, i], np_syn[:, i]]
        if key in df_tom:
            temp = df_tom[key].dropna()
            data.append(temp)

            # fill with colors - overwrite list of colors
            boxcolors = ['lightgreen', 'seagreen', 'lightblue']
            x_ticks_labels = [f"real", f"syn",  f"tom"]

        ax = plt.subplot(2, int(np.ceil(nr_props/2)), i + 1)

        # Creating plot
        # plt.boxplot(data, patch_artist = True, boxprops = dict(facecolor = colors))
        B = ax.boxplot(data, patch_artist=True, notch=False,
                       medianprops={"color": "black", "linewidth": 1.0})

        # give boxes different colors
        for patch, color in zip(B['boxes'], boxcolors):
            patch.set_facecolor(color)

        plt.ylabel(use_units[i])
        ax.set_title(metrics_caps[i])
        ax.set_xticklabels(x_ticks_labels)
        
        # assign your quartiles, limits and iq3
        q1, q3 = np.percentile(data[1], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        #create conditions to isolate the outliers
        outliers = data[1][(data[1] < lower_bound) | (data[1] > upper_bound)]
        max_prefix_length = max(len(cap.split(":")[0]) for cap in metrics_caps)
        max_prefix_length = max_prefix_length + 10
        print("Numer of outliers of ", metrics_caps[i], " : ", (len(outliers)) )
        

    bbox_file = os.path.abspath(plot_path + "/" + f"{name}_box.png")
    plt.savefig(bbox_file, dpi=300)

    return


def histogram_metrics(np_org, np_syn, df_tom, use_metrics, use_units, metrics_caps,
                      nr_props, plot_path, stats_path, name):
    '''
    Create histogram of distribution of metrics for original (real), synthetic, 
    and tomtec data.

    Parameters
    ----------
    np_org : (N, nr_props) numpy nd array
        Array of metrics for N real data shapes and nr_props properties.
    np_syn : (M, nr_props) numpy nd array
        Array of metrics for M synthetic data shapes and nr_props properties.
    df_tom : pandas.Dataframe
        Dataframe with TomTec metrics for N real data shapes and P TomTec result
        properties.
    use_metrics : dictionary
        Dictionary of all metrics to use.
    use_units : list
        List with strings of the metric units.
    metrics_caps : list
        List with strings of the metric captions.
    nr_props : integer
        Number of properties / metrics to evaluate.
    plot_path : string
        Absolute path to the plot directory.
    stats_path : string
        Absolute path to the statistics directory.
    name : string
        Name of the data, e.g. the state (diastolic or systolic).

    Returns
    -------
    None.

    '''

    print(f"\nPlot histogram of metrics for {name}...")
    # plot settings
    # inner size structure of figures
    top = 0.95
    bottom = 0.055
    left = 0.05
    right = 0.98
    hspace = 0.25
    wspace = 0.8
    # fonts
    font = {'size': 8}

    # parameters for histogram
    kwargs = dict(histtype='step', alpha=0.9, density=True)

    # subplot for each property
    fig = plt.figure(figsize=(11, 7))
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        hspace=hspace, wspace=wspace)

    data = []
    p_welch = []
    for i, key in enumerate(use_metrics.keys()):
        # allocate data for boxplot
        data = [np_org[:, i], np_syn[:, i]]

        ax = plt.subplot(2, int(np.ceil(nr_props/2)), i + 1)

        # shapiro wilk test
        stats_org, p_org = scp.stats.shapiro(np_org[:, i])
        stats_syn, p_syn = scp.stats.shapiro(np_syn[:, i])
        txt = [str(round(p_org, 2)), str(round(p_syn, 2))]

        # Creating plot
        plt.hist(np_org[:, i], **kwargs)
        plt.hist(np_syn[:, i], **kwargs)

        # ax[i].legend()
        # plt.legend()
        # fig.legend()

        plt.xlabel(use_units[i])
        # plt.ylabel("Density")
        ax.set_title(metrics_caps[i])

        # welch t test
        stats_welch, p_welch_i = scp.stats.ttest_ind(
            np_org[:, i], np_syn[:, i], equal_var=False)
        p_welch.append(p_welch_i)

        print(f"Welch t-test of '{metrics_caps[i]}' :   p = ", round(p_welch_i, 3))

    plt.legend([f"{name} real", f"{name} syn"], bbox_to_anchor=(2.3, 0.5), ncol=1)
    histo_file = os.path.abspath(plot_path + "/" + f"{name}_histo_shapiro_test.png")
    plt.savefig(histo_file, dpi=300)

    # write p_welch test to file
    p_welch_file = os.path.join(stats_path, f"p_welch_{name}.txt")
    np.savetxt(p_welch_file, p_welch)

    return


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
    sns.pairplot(df_data)

    correl_file = os.path.abspath(plot_path + "/" + f"{name}_{data}_correlation.png")
    plt.savefig(correl_file, dpi=300)

    return


def plot_correlation_colored_by_key(np_data_list, metrics_caps, plot_path, key, names, data):
    '''
    Plot the states correlation matrix between all parameters.

    Parameters
    ----------
    np_org : list of arrays, (N, nr_props) numpy nd array
        Lists of arrays of metrics for N data shapes and nr_props properties.
    metrics_caps : list
        List with strings of the metric captions.
    plot_path : string
        Absolute path to the plot directory.
    key : string
        Key/header for pandas dataframe to create.
    names : list of strings
        Name of the data, e.g. the state (diastolic or systolic).
    data : string
        Category of data, e.g. real or synthetic ('real' or 'syn').

    Returns
    -------
    None.

    '''
    print(f"\nPlot states correlation matrix of metrics for {data}...")

    df_list = []

    for count, np_data in enumerate(np_data_list):
        # basic correlation between metrics of data
        df_data = pd.DataFrame()

        # import np data to dataframe
        for i in range(len(metrics_caps)):
            df_data[metrics_caps[i]] = np_data[:, i]

        # add state to dataframe
        df_data[key] = names[count]

        # add to list
        df_list.append(df_data)

        
    # append all dataframes
    df_all = df_list[0].append(df_list[1:])
    df_all = df_all.reset_index()
    df_all = df_all.drop('index', axis=1)

    # pairplot correlation basic matrix
    sns.pairplot(df_all, hue=key)

    correl_file = os.path.abspath(plot_path + "/" + f"{key}_{data}_correlation.png")
    plt.savefig(correl_file, dpi=300)

    return


def plot_states_scatter(m_org_dia, m_org_sys, m_syn_dia, m_syn_sys, 
                        metrics_caps, plot_path, rows=4):
    '''
    Scatter plot of features/metrics between states.

    Parameters
    ----------
    m_org_dia : list of arrays, (N, nr_props) numpy nd array
        Lists of arrays of metrics for N data shapes and nr_props properties for the original diastolic data.
    m_org_sys : list of arrays, (N, nr_props) numpy nd array
        Lists of arrays of metrics for N data shapes and nr_props properties for the original systolic data.
    m_syn_dia : list of arrays, (N, nr_props) numpy nd array
        Lists of arrays of metrics for N data shapes and nr_props properties for the synthetic diastolic data.
    m_syn_sys : list of arrays, (N, nr_props) numpy nd array
        Lists of arrays of metrics for N data shapes and nr_props properties for the synthetic systolic data.
    metrics_caps : list
        List with strings of the metric captions.
    plot_path : string
        Absolute path to the plot directory.

    Returns
    -------
    None.

    '''
    # limit metric size to metrics used for the analysis
    # note : if metric files have different size then the metrics_caps (set in the infile.py)
    m_org_dia = m_org_dia[:, :len(metrics_caps)]
    m_org_sys = m_org_sys[:, :len(metrics_caps)]
    m_syn_dia = m_syn_dia[:, :len(metrics_caps)]
    m_syn_sys = m_syn_sys[:, :len(metrics_caps)]
    
    print(f"\nPlot state to state scatter plots of metrics...")

    # inner size structure of figures
    top = 0.95
    bottom = 0.055
    left = 0.05
    right = 0.98
    hspace = 0.25
    wspace = 0.8
    
    # fonts
    font = {'size': 6}
    plt.rc('font', **font)
    
    # SMALL_SIZE = 8
    # MEDIUM_SIZE = 10
    # BIGGER_SIZE = 12
    
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # subplot for each property
    fig = plt.figure(figsize=(11, 7))
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        hspace=hspace, wspace=wspace)
    
    for i in range(len(metrics_caps)):
        # subplots scatter1
        ax = plt.subplot(rows, int(np.ceil(len(m_org_dia[0,:])/rows)), i + 1)
        
        ax.scatter(m_org_dia[:,i], m_org_sys[:,i], s=6) #, label="org")
        ax.scatter(m_syn_dia[:,i], m_syn_sys[:,i], s=6) #, label="syn")
        
        # metrics titles
        ax.set_title(metrics_caps[i])
        
        plt.xlabel("dia")
        plt.ylabel("sys")
        
        # get spearman correlation r value
        r_org, _ = scp.stats.spearmanr(m_org_dia[:,i], m_org_sys[:,i])
        r_syn, _ = scp.stats.spearmanr(m_syn_dia[:,i], m_syn_sys[:,i])
        
        # labels
        label_org = f"r_org={round(r_org,2)}" 
        label_syn = f"r_syn={round(r_syn,2)}"
        
        plt.legend([label_org, label_syn], loc="lower right")

    # plt.legend(['org','syn']) #, bbox_to_anchor=(.95, -0.15), ncol=2)
    
    plt.tight_layout()
    correl_file = os.path.abspath(plot_path + "/" + f"state_to_state_correlation.png")
    plt.savefig(correl_file, dpi=300)

    return



def plot_correlation_heatmap(np_data, metrics_caps, plot_path, name, data, corr_method="pearson"):
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
    corr_method : string
        Correlation coefficient method, e.g. pearson, spearman, kendall etc.
        Default is pearson.

    Returns
    -------
    None.

    '''

    print(f"\nPlot spearman heatmap of metrics for {name}_{data}...")
    # spearman correlation heatmap between metrics of original data

    # basic correlation between metrics of data
    df_data = pd.DataFrame()

    # import np data to dataframe
    for i in range(len(metrics_caps)):
        df_data[metrics_caps[i]] = np_data[:, i]

    # file name to save plot
    spear_file = os.path.abspath(
        plot_path + "/" + f"{name}_{data}_{corr_method}.png")

    r = display_correlation(df_data, spear_file)

    return r
