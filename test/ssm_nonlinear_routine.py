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

from tools.io_utils import *
from ssm.ssm_linear import *
from tools.pointclouds import *
from tools.plots import *
from tools.io_utils import *

from morphomatics.geom import Surface
from morphomatics.stats import StatisticalShapeModel
from morphomatics.manifold import FundamentalCoords
from morphomatics.manifold import DifferentialCoords
from morphomatics.manifold import PointDistributionModel

###############################################################################

work_dir = os.path.abspath("/data/")

save_folder = "nonlin_LV"
save_dir = os.path.join("/home/user/Documents/data_ssm/", save_folder)


# number of modes to plot
n_modes = 3

os.chdir(work_dir)
###############################################################################
# Initialization for Morphomatics Model
###############################################################################

LV_dir = "echo_LV_org"
meshes = read_data_morphomatics(LV_dir)



##########################################################################
# Construct SSM - Morphomatics Fundamental coordinates
##########################################################################
# to Surface type
# function to read meshes
# NOTE for reshaping: first column of faces is filled with int 3 - can be cut out
as_surface = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
surfaces_dia = [as_surface(m) for m in meshes]


print("\n--- Computation of SSM via fundamental coordinates ---\n")
t4 = time.time()
# construct model with fundamentalCoords function - use non-linear SSM model
SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
SSM.construct(surfaces_dia)

t5 = time.time()
print("\nTime nonlinear SSM    ", round(t5-t4, 5), " seconds")

# standard deviation associated to the modes; std_dev is sqrt of variance
var = SSM.variances
std = np.sqrt(SSM.variances)

print("Length of MV (non-linear)  ", var.shape)
print("8 entry var of MV (non-linear)  ", var[7])

# variances for SSM - compare linear and non-linear model
print("\nLV : Variances for linear and non-linear SSM")
for i in range(len(std)):
    print(f"Relative part of {i+1}. mode         ",  round(S_var[i]/np.sum(S_var)*100,2),            "%", "      ", round(var[i]/np.sum(var)*100,2), "%")
    print(f"Cumulative variance for {i+1} modes  ",  round(np.sum(S_var[:i+1])/np.sum(S_var)*100,2), "%", "      ", round(np.sum(var[:i+1])/np.sum(var)*100,2) , "%")
    # dia
    nonlin_var.append(S_var[i] / np.sum(S_var) * 100)
    nonlin_var_sum.append(np.sum(S_var_MV[:i+1])/np.sum(S_var)*100)


##########################################################################
# Construct SSM - Morphomatics PointDistribution Model
##########################################################################
print("\n--- Computation of SSM via PointDistribution Model ---\n")
# construct model with PointDistributionModel function - linear Model
SSM_point = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
SSM_point.construct(surfaces_dia)

std_point = np.sqrt(SSM_point.variances)


##########################################################################
# Construct SSM - Morphomatics Differential Coords
##########################################################################
print("\n--- Computation of SSM via differential coordinates ---\n")
# construct model with PointDistributionModel function - linear Model
SSM_diff = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))
SSM_diff.construct(surfaces_dia)

std_diff = np.sqrt(SSM_diff.variances)


##########################################################################
# Plotting variance
##########################################################################
fig = plt.figure()
font = {'size'   : 40}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('font', family='mathptmx') # mathptmx   # serif

fig.subplots_adjust(
    top=top,
    bottom=bottom,
    left=left,
    right=right,
    hspace=hspace,
    wspace=wspace
)

print("length of non-lin var", len(nonlin_var))
plt.plot(range(1,len(nonlin_var)+1), nonlin_var, label="non-linear", marker='o', markersize=14)

# plt.xticks(range(1,len(S_ev)) ) #, fontsize=20)
plt.ylabel(r"variance (\%)")
plt.xlabel(r"modes")
plt.ylim(0, 45)  #70
plt.legend() #loc="upper left")

plt.grid()
plt.show()


##########################################################################
# Nonlinear Fundamental Coords (Morphomatics)
##########################################################################
pv.global_theme.show_edges = True
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

pl = pv.Plotter(shape=(n_modes,4))
pl.link_views()

morph_modes = []

# nonlinear model
k=0
for i in range(n_modes):
    for j, t in enumerate(np.linspace(-3,3,2)):
        # excite mode
        coords = SSM.space.exp(SSM.mean_coords, t * std[i] * SSM.modes[i]) # t* std
        # map shape space coords to vertex coords
        v = SSM.space.from_coords(coords)
        # add vertices to array
        morph_modes.append(v)
        # add mesh to plot
        pl.subplot(i, j)
        pl.add_mesh(pv.PolyData(v, meshes[0].faces))
        pl.view_xz()
        pl.camera.roll += 180
        pl.camera.zoom(1)
        # plot txt
        txt1 = "nonlin"
        txt2 = f"\nmode {i+1}\n" + f"b = {t}"
        txt = txt1 + txt2
        pl.add_text(txt, position='upper_left', color='black', font_size=8)
        k += 1

        # polar plot of mode
        # compute mean shape of non linear SSM
        v_mode = SSM.space.from_coords(t * std[i] * SSM.modes[i])
        print(v_mode.shape)
        print(v_mode)
        segments = create17segments(v_mode)
        z = averageOnSegments(v_mode, segments)
        name = f"dia_only_mode{i+1}_std{t}.png"
        save_plot = os.path.join(save_dir, name)
        min_val = -10
        max_val = 100
        plot17segments(z, save_plot, "mode excitation (mm)", min_val, max_val)

        # write mesh
        save_file = os.path.join(save_dir, f"dia_only_mode{i+1}_std{t}.stl")
        # writeTriangularSurfaceMesh(v, faces, save_file)


save_file = os.path.join(save_dir, "dia_compare_SSM_linear_nonlinear.png")
# pl.show(screenshot=save_file, window_size=[1800,1110])
