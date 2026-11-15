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


def read_real_metrics(metric_path, name):
    
    org_file = os.path.join(metric_path, f"metrics_{name}_original.txt")
    
    m_org = np.loadtxt(org_file)
    
    return m_org


def read_synthetic_metrics(metric_path, name, ext=""):
    
    syn_file = os.path.join(metric_path, f"metrics_{name}_synthetic{ext}.txt")
    print(f"metrics_{name}_synthetic{ext}.txt")
    m_syn = np.loadtxt(syn_file)
    
    print("Length of m_syn ", len(m_syn))
    
    return m_syn


def read_tomtec_evaluation(tomtec_file, nr_props, name):
    # tom tec result data
    tomtec = pd.read_csv(tomtec_file)
    # remove special charaters
    tomtec.columns = tomtec.columns.str.replace("[°,#]", "")

    # initialize matrix with same setup as Metrics Dictionary
    m_tomtec = np.array((len(tomtec.index), nr_props))

    option = [f"{name}"]
        
    # selecting rows based on condition 
    df = tomtec[tomtec['state'].isin(option)]

    # allocating panda (csv) properties to structure of own-defined python metrics
    m_tom = {}

    m_tom_mean = {}
    m_tom_std = {}

    # ci_annulus
    m_tom["ci_annulus"] = df["RESULT_ANNULUS_SPHERICITY_INDEX[cm/cm]"]

    # circ_annulus - [cm] to [mm]
    m_tom["circ_annulus"] = df["RESULT_ANNULUS_CIRCUMFERENCE[cm]"]*10

    # A_annulus - [cm]*[cm] to [mm]*[mm]
    m_tom["A_annulus"] = df["RESULT_ANNULUS_AREA_2D[cm*cm]"]*10*10

    # h_annulus - [cm] to [mm]
    m_tom["h_annulus"] = df["RESULT_ANNULUS_HEIGHT[cm]"]*10

    # A_lf_ant - [cm]*[cm] to [mm]*[mm]
    m_tom["A_lf_ant"] = df["RESULT_LEAFLETS_ANTERIOR_LEAFLET_AREA[cm*cm]"]*10*10

    # len_lf_ant - [cm] to [mm]
    m_tom["len_lf_ant"] = df["RESULT_LEAFLETS_ANTERIOR_LEAFLET_LENGTH[cm]"]*10

    # len_lf_post - [cm] to [mm]
    m_tom["len_lf_post"] = df["RESULT_LEAFLETS_POSTERIOR_LEAFLET_LENGTH[cm]"]*10

    # A_lf_post - [cm]*[cm] to [mm]*[mm]
    m_tom["A_lf_post"] = df["RESULT_LEAFLETS_POSTERIOR_LEAFLET_AREA[cm*cm]"]*10*10

    # ang_lf_post
    m_tom["ang_lf_post"] = df["RESULT_LEAFLETS_POSTERIOR_LEAFLET_ANGLE[]"]*(-1)

    # ang_lf_ant
    m_tom["ang_lf_ant"] = df["RESULT_LEAFLETS_DISTAL_ANTERIOR_LEAFLET_ANGLE[]"]*(-1)

    return m_tom