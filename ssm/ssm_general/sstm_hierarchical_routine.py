################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Hierarchical model using non-linear shape space.
Implementation with Morphomatics Version 3.0 (currently unpublished).

Create dependent SSM model for multiple state e.g. diastolic and systolic state of LV/MV
Non-linear geodesic interpolation between the states.

Script includes:
    - pyvista interactive animation
    - write selected cohort of interpolated and/or excited modes
    - run iterative closest point algorithm to align data and compare with linear results
    - translate all geometries to center of mass

'''

import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

from typing import List, NamedTuple
import pyvista as pv

import numpy as np
import jax.numpy as jnp

from morphomatics.geom import Surface
from morphomatics.manifold import FundamentalCoords, DifferentialCoords
from morphomatics.manifold import TangentBundle, ShapeSpace
from morphomatics.stats import PrincipalGeodesicAnalysis as PGA
from morphomatics.stats import StatisticalShapeModel

from funcs.io_utils import *
from funcs.pointclouds import *
from funcs.LV_preproc import *
from funcs.plots import *
import matplotlib.pyplot as plt
############################################################
# Read data
############################################################
# folder_name = "echo_MV_test"
folder_name = "echo_LV_org"    # org

data_dir = os.path.abspath("/data/")
path = os.path.join(data_dir, folder_name)

# set flag for LV/MV to plot volume
comp_vol = False

save_folder = "nonlin_LV/"
save_folder_inter = "interpolation_LV"
aligned = "nonlin_LV_aligned"
save_folder_ex = "nonlin_LV_ex"

# save_folder = "nonlin_MV/"
# aligned = "nonlin_MV_aligned"
# save_folder_inter = "interpolation_MV"

save_dir = os.path.abspath("/home/user/Documents/data_ssm/")
save_path = os.path.join(save_dir, save_folder)
save_inter = os.path.join(save_dir, save_folder_inter)
save_ex = os.path.join(save_dir, save_folder_ex)

read_msh = os.path.join(data_dir, "echo_LV_test/18/B4_18_LV_BeutelRevision+LVOT_007_transMV.stl")
# read_msh = os.path.join(data_dir, "echo_MV_test/18/18_MV_diast_new_010_transMV.stl")


# number of modes to save
n_modes = 5
# how many interpolation time steps: n_time = 1 >> only state 0 (sys) and state 1 (dia)
n_time = 1
n_int = 5

# excitation
ex = 3

# plotting
top = 0.98
bottom = 0.111
left = 0.1
right = 0.985
hspace = 0.2
wspace = 0.2

############################################################
############################################################
class Subject(NamedTuple):
    systole: Surface
    diastole: Surface

# create Surface from pv.PolyData
to_surf = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])

subjects: List[Subject] = []
for d in os.listdir(path):
    files = os.listdir(f'{path}/{d}')
    files.sort()
    sub = Subject(*[to_surf(pv.read(f'{path}/{d}/{f}')) for f in files])
    subjects.append(sub)

############################################################
# Determine reference shape & setup shape space
############################################################

# construct SSM for diastole configurations
SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
SSM.construct([s.diastole for s in subjects])
print(f'variances: {SSM.variances}')

# plotting variance
fig = plt.figure()
font = {'size'   : 28}

plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('font', family='mathptmx')

fig.subplots_adjust(
    top=top,
    bottom=bottom,
    left=left,
    right=right,
    hspace=hspace,
    wspace=wspace
)

# eigenvalues for complete SSM
var = SSM.variances
sum_var = []
rel_var = []
for i in range(len(var)):
    rel_var = rel_var + [var[i]/np.sum(var)*100]
    sum_var = sum_var + [np.sum(var[:i+1])/np.sum(var)*100]


plt.plot(range(1,len(var)+1), sum_var, label="total variance", marker='o', markersize=14)
plt.plot(range(1,len(var)+1), rel_var, label="relative variance", marker='o', markersize=14)
plt.hlines(95, xmin=1, xmax=len(var), color='red', linestyles='--', label='95% threshold')

plt.xticks(range(1,len(var)+1) ) #, fontsize=20)
plt.ylabel(r"total variance (\%)")
plt.xlabel(r"modes")
plt.legend() #loc="upper left")

manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
manager.window.showMaximized()
plt.grid()
plt.show()

# write2file
var_file = os.path.join(save_path, 'var_nonlin_LV.log')
with open(var_file, 'w') as f:
    for i in range(len(var)):
        f.write(str(rel_var[i]) + '\n')

############################################################
# Construct statistical shape trajectory model (SSTM)
############################################################
print('construct SSTM')
M: ShapeSpace = SSM.space
TM = TangentBundle(M)

# identify start- and end-points with tangent vectors/geodesics in shape space
pts2vec = lambda p, q: [p, M.connec.log(p, q)]
data = jnp.array([pts2vec(M.to_coords(s.diastole.v), M.to_coords(s.systole.v)) for s in subjects])

# compute mean and main modes of variation
pga = PGA(TM, data)

############################################################
# Visualize SSTM
############################################################
print('visualize SSTM')
pv.global_theme.show_edges = True
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'


# pyvista meshes for mean trajectory and one that is varied by principal mode
nu = pv.PolyData(np.asarray(SSM.mean.v), np.c_[np.full(len(SSM.mean.f), 3), SSM.mean.f])
var = nu.copy()

# compute shape at time t given trajectory (i.e. tangent vector) p,v
exp = lambda p, v, t: M.from_coords(M.connec.exp(p, t*v))

pl = pv.Plotter(shape=(1,2))
pl.subplot(0, 0)
pl.add_title(r'mean trajectory')
pl.add_mesh(nu)

pl.subplot(0, 1)
pl.add_title(r'principal mode')
pl.add_mesh(var)
pl.link_views()



# init excited trajectory as mean one
excited = pga.mean

# interactive parameters
t = 0; s = 0; i = 0

# slider widget position
pos = lambda x: {'pointa': (x, .1), 'pointb': (x+.3, .1)}

def update_t(t_):
    global t
    t = t_
    nu.points = exp(*pga.mean, t)
    var.points = exp(*excited, t)
pl.add_slider_widget(callback=update_t, title='time', rng=[0.,1.], value=t, style='modern', **pos(.03))

def update_amplitude(s_):
    global s, excited
    s = s_
    dev = np.sqrt(pga.variances[i])
    excited = TM.connec.exp(pga.mean, s * dev * pga.modes[i])
    var.points = exp(*excited, t)
pl.add_slider_widget(callback=update_amplitude, title='amplitude', rng=[-3.,3.], value=s, style='modern', **pos(.35))

def update_index(i_):
    global i
    if int(i_) == i:
        return
    i = int(i_)
    update_amplitude(s)

#pl.add_slider_widget(callback=update_index, title='index', rng=[0,len(subjects)], value=i, style='modern', **pos(.67))
#pl.show()


############################################################
# save modes and interpolation geometries
############################################################
# get faces
vertices, faces = readTriangularSurfaceMesh(read_msh)

# volume init
faces_pv = np.concatenate((3 * np.ones(faces.shape[0])[:, np.newaxis], faces), axis=1)
faces_pv = np.array(faces_pv.flatten(), dtype=int)

dia_vol = []
sys_vol = []

# segments and values of mean shape
# np.asarray(SSM.mean.v)
# iteration over modes
for i in range(n_modes):
    # iteration over positive and negative excitation
    for j, s in enumerate(np.linspace(-ex, ex, 2)):
        for k in range(n_time+1):
            # time interpolation parameter
            t  = k / n_time
            # update parameters
            dev = np.sqrt(pga.variances[i])
            excited = TM.connec.exp(pga.mean, s * dev * pga.modes[i])
            var.points = exp(*excited, t)

            print("\ns ", s, ",  i ", i, ",  t ", t)
            # print("\nvar ", var.points)
            if t==0.0:
                pre = "dia"
                # compute volume if flag is true - only for LV
                if comp_vol == True:
                    poly = pv.PolyData(var.points.flatten(), faces_pv)
                    vol = poly.volume
                    dia_vol = dia_vol + [vol]

            elif t==1.0:
                pre = "sys"
                # compute volume if flag is true - only for LV
                if comp_vol == True:
                    poly = pv.PolyData(var.points.flatten(), faces_pv)
                    vol = poly.volume
                    sys_vol = sys_vol + [vol]

            else:
                pre = t
            # msh = f"t{t}_mode{i}_std{s}.stl"
            msh = f"{pre}_both_mode{i+1}_std{s}.stl"

            # polar plot of mode
            # compute mean shape of non linear SSM
            '''
            segments = create17segments(var.points)
            z = averageOnSegments(var.points, segments)

            name = f"{pre}_both_mode{i+1}_std{s}.png"
            save_plot = os.path.join(save_path, name)
            min_val = -10
            mx_val = 40
            plot17segments(z, save_plot, "mode excitation (mm)", min_val, max_val)
            '''
            # write mesh
            save_msh = os.path.join(save_path, msh)
            ## print("Save mesh dir  ", save_msh)
            ## writeTriangularSurfaceMesh(var.points, faces, save_msh)

print("\nEnd of mode computation !\n")

############################################################
# interpolation of mean shape between dia/sys
############################################################
# get faces
vertices, faces = readTriangularSurfaceMesh(read_msh)

# volume init
faces_pv = np.concatenate((3 * np.ones(faces.shape[0])[:, np.newaxis], faces), axis=1)
faces_pv = np.array(faces_pv.flatten(), dtype=int)



for k in range(n_int+1):
    # time interpolation parameter
    # time interpolation parameter
    t  = k * 1/(n_int)

    # update parameters : no excitation
    excited = TM.connec.exp(pga.mean, 0 * pga.mean)
    var.points = exp(*excited, t)

    print("\ns ", 1, ",  i ", i, ",  t ", t)

    msh = f"t{t}_mean.stl"

    # write mesh
    save_msh = os.path.join(save_inter, msh)
    ## print("Save mesh dir  ", save_msh)
    ## writeTriangularSurfaceMesh(var.points, faces, save_msh)


############################################################
# dia other excitations of non-linear SSM
############################################################

# mode 1
i = 0
# diastolic state
t = 0

for s in range(1,6):
    # update parameters
    dev = np.sqrt(pga.variances[i])
    excited = TM.connec.exp(pga.mean, s * dev * pga.modes[i])
    var.points = exp(*excited, t)

    print("\ns ", s, ",  i ", i, ",  t ", t)

    msh = f"mode1_dia_s{s}.stl"

    # write mesh
    save_msh = os.path.join(save_ex, msh)
    print("Save mesh dir  ", save_msh)
    writeTriangularSurfaceMesh(var.points, faces, save_msh)

############################################################
# plot volume over modes
############################################################
modes =  []
for i in range(int(len(sys_vol)/2)):
    min_mode = f"min{i+1}"
    max_mode = f"max{i+1}"
    modes = modes + [min_mode, max_mode]

if comp_vol == True:
    print("Plotting volume")
    font = {'size'   : 20}
    plt.rc('font', **font)

    plt.figure()

    x = range(len(dia_vol))

    plt.plot(x, dia_vol, label="dia", marker='o')
    plt.plot(x, sys_vol, label="sys", marker='o')
    # plt.axhline(y = mean_volume[i], color='grey', linestyle = '--', alpha=0.5)

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xticks(x, modes, fontsize=16)
    plt.legend(loc="lower right")
    plt.ylabel("LV volume")
    plt.show()

############################################################
# alignment of modes via landmarks
############################################################

# see align_data.py


'''
############################################################
# run icp
############################################################
from funcs.pointclouds import *

# IMPORTANT : move template geometry to directory before running the icp
# template geometry has same orientation as data to compare to
# ideally : aligned and centered
# rename it as "template" - in the beginning of filename "0_" to have file at list index 0

aligned_path = os.path.join(save_dir, aligned)

# find index
# files = os.listdir(save_path)

# set index manually
print(save_path)
idx_target = 0
iterative_closest_point(save_path, idx_target, aligned_path)


############################################################
# translate to center again
############################################################
os.chdir(aligned_path)

files = os.listdir()
files = select_files_with_extension(files, "stl")

for file in files:
    vertices, faces = readTriangularSurfaceMesh(file)
    ctr = vertices.mean(axis=0)
    vertices_moved = vertices - ctr #np.expand_dims(ctr, axis=1)
    writeTriangularSurfaceMesh(vertices_moved, faces, file)

'''