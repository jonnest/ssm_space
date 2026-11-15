################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Atlas construction with Deformetrica for multiple objects and with multiple iterations

@requirements: file structure to use this file - only /data folder required, the rest will be created
    /Case_dir
        /data       :   (REQUIRED) .stl files aligned and centered for atlas construction

        /data_vol   :   if open geometries, saved cloesd geometries for computing volume
        /output0    :   copied .stl files from /data with modified names for iteration algorithm
        /output1    :   iteration nr {1} of atlas reconstruction
        /..         :   depending on how many iterations (nr_it) of deformetrica, number of output folders

@input:
	- data_path : path of the main directory
	- overall_caseName : name of case directory
    - target_nr : number of desired nodes for reconstructed meshes
	- def_it : number of iterations within the Deformetrica algorithm
	- nr_it : number of outer iterations of the Deformetrica algorithm - for normal reconstruction nr_it=1


@status:
    [] update header for general usage

'''
import os.path
import shutil
import glob
import deformetrica as dfca
import time
import psutil

from tools.pymeshlab import *
from tools.io_utils import *

###############################################################################
#                           Initialization
###############################################################################
workpath = "C:/Users/user/Documents/10_MV_SSM/deformetrica_test"

overall_caseName = "mv_atlas_v1"

# type of shape/geometry : choose between "LV" (left ventricle / closed geometry)
# and "MV" (mitral valve / open geometry)
# LV : template chosen via mean volume
# MV : template chosen via mean diagonal of bounding box
geo_type = "MV"

# set paths
data_path = os.path.abspath(workpath + "/" + overall_caseName + "/")
data_base = os.path.join(data_path, 'data/')
# is generated automatically
data_vol = os.path.join(data_path, 'data_vol/')

# meshing : target number of nodes for reconstructed geometries; e.g. 500, 750, 1000, 2000, 3000, 4000, 5000
target_nr = 500

# number of iterations for recomputing mean shape and running deformetrica
# if tolerance is reached then iterations will stop earlier
# def_it : 3000
def_it = 3000
# nr_it : default for normal reconstruction = 1
nr_it = 1

# choice of kernel size dependent on template
# percentage of bounding's box diagonal of template, default = 0.1
per = 0.1

# set bool flag to close geometry: yes=True, no=False, default = False
close_geometry = False

# exponent of tolerance for estimator, default = -8
tol_exp = -8

###############################################################################
#                            Preprocessing
###############################################################################
# Check whether the data_vol exists or not
if not os.path.exists(data_vol) and geo_type == "LV":
   os.makedirs(data_vol)

# find template for first iteration of atlas construction
# LV : volume array -- find mean
# MV : diagonal of bbox -- find mean
# read meshes and close geometries to get volume
os.chdir(data_base)
files = os.listdir()
stl_files = select_files_with_extension(files, '.stl')
stl_files = sorted(stl_files)
S = np.empty(len(stl_files))

if geo_type == "LV":
    j = 0
    for stl_file in stl_files:
        # close holes for computing volume and remeshing
        close_all_holes(stl_file, data_vol + stl_file)

        if close_geometry == True:
            print("--------------------------------")
            print("Remeshing closed geometry nr ", j)
            print("--------------------------------")
            isotropic_remeshing_target(data_vol + stl_file, target_nr)
            # set data base to closed volumes
            data_base = data_vol

        vertices, faces = readTriangularSurfaceMesh(data_vol + stl_file)
        # create polydata
        faces_pv = np.concatenate((3 * np.ones(faces.shape[0])[:, np.newaxis], faces), axis=1)
        faces_pv = np.array(faces_pv.flatten(), dtype=int)
        mesh = pv.PolyData(vertices, faces_pv)

        vol = mesh.volume
        S[j] = vol
        j = j + 1

elif geo_type == "MV":
    j = 0
    for stl_file in stl_files:
        # get bounding box of mesh
        bbox = measure_mesh(stl_file)
        # get diagonal of bounding box
        diag = bbox.diagonal()
        S[j] = diag
        j = j + 1

# redefine data base if "LV"
os.chdir(data_base)
files = os.listdir()
stl_files = select_files_with_extension(files, '.stl')
stl_files = sorted(stl_files)

# check if template file already exists
for stl_file in stl_files:
    if 'Template_' in stl_file:
        os.remove(stl_file)
        stl_files.remove(stl_file)

# compute mean volume
S_mean = np.mean(S)
min_val = min(abs(S - S_mean))
S_diff = S - S_mean
print("------------------------------")
print("Mean value ", S_mean)
print("------------------------------")
# find index of item closest to mean volume of S list
idx_mean = np.where(abs(S_diff) == min_val)
idx_mean = idx_mean[0][0]

# determine template case
template_case = stl_files[idx_mean]

# remeshing with target as number of vertices
print("-----------------------")
print("Remeshing template file")
print("-----------------------")
remeshed_file = isotropic_remeshing_target(template_case, target_nr, 'Template_')

# STL2VTK
filename = remeshed_file.split('.')[0]
mesh = pv.read(remeshed_file)
mesh.save(filename + ".vtk",binary=False)
template_vtk = filename + '.vtk'

# 1) do atlas construction of mean volume shape
#    remeshing isotropic in Meshlab -- to appropriate vertices number ~2000
# 2) do atlas construction of mean shape of node correspondence shapes

# compute kernel width
bbox = measure_mesh(remeshed_file)
kernel_width = per * bbox.diagonal()
print("Diagonal of bounding box  ", round(bbox.diagonal(),3))
print("\nKernel width for template and deformation is  ", round(kernel_width,3), "\n")

###############################################################################
#                              Deformetrica
###############################################################################

# starting time
t = np.empty(nr_it+1)
t[0] = time.time()

os.chdir(data_base)

# STL2VTK
stl_files = glob.glob(data_base + "*.stl")
for stl_file in stl_files:
    filename = stl_file.split('.')[0]
    mesh = pv.read(stl_file)
    mesh.save(filename + ".vtk", binary=False)
vtk_files = glob.glob(data_base + "*.vtk")

vtk_files = list(filter(lambda x: x != data_base + template_vtk, vtk_files))

# estimator callback to save and later plot the log-likelihood values
iteration_status_dictionaries = []

def estimator_callback(status_dict):
    iteration_status_dictionaries.append(status_dict)
    return True

# Set the parameters using python dictionnaries
# name of the object
object_name = 'LV'

# init step - copy data files to output0 - zero iteration == dataset
# create the output dir if doesn't exists and delete the previous files
data_out0 = os.path.join(data_path, 'output0/')
if os.path.exists(data_out0) and os.path.isdir(data_out0):
    shutil.rmtree(data_out0)

shutil.copytree(data_base, data_out0)
os.chdir(data_out0)
files = os.listdir()
for file in files:
    os.rename(file, 'Reconstruction_' + file)

###############################################################################
# dataset init
###############################################################################
data_files = []  # list of the data, without the template
sub_ids = []  # list of filename
for data_file in vtk_files:
    dict_data = {object_name: data_file}
    data = [dict_data]
    data_files.append(data)
    sub_id = data_file.split('/')[-1].split('.')[0]
    sub_ids.append(sub_id)

# set the data, model, parameters dictionaries
# choose only one
dataset_specifications = {
    'dataset_filenames': data_files,
    'subject_ids': sub_ids
}

###############################################################################
# multiple iterations to get better node correspondence
# results are not significant
###############################################################################
for j in range(nr_it):
    # new data base
    data_base = os.path.join(data_path, f'output{j}/')
    data_out = os.path.join(data_path, f'output{j+1}/')

    # choice of the template - use mean volume
    os.chdir(data_base)
    if j == 0:
        template_path = template_vtk
        template_path = 'Reconstruction_' + template_path
        print('Initialization step')
        print('Template path', template_path)
    elif j != 0:
        # read reconstruction meshes
        meshes = os.listdir()
        nr_shapes = 0
        for mesh in meshes:
            if 'Template_' in mesh and 'stl' in mesh:
                vertices, faces = readTriangularSurfaceMesh(mesh)
            if 'Reconstruction_' in mesh and 'stl' in mesh:
                nr_shapes += 1
        print('\nNumber of shapes  ', nr_shapes, '  \n')
        # add number of shapes in reconstruction - in loop style take init information
        shapes = np.empty((len(vertices) * 3, nr_shapes))
        i = 0
        for mesh in meshes:
            if 'Reconstruction_' in mesh and 'stl' in mesh:
                vertices, faces = readTriangularSurfaceMesh(mesh)
                vertices_1d = np.array(np.reshape(vertices, (len(vertices) * 3, 1)))
                shapes[:, i] = vertices_1d[:, 0]
                i += 1

        # compute mesh shape
        os.chdir(data_base)
        mean_shape = shapes.mean(axis=1)
        vertices_mean = mean_shape.reshape(int(len(mean_shape) / 3), 3)
        mean_name = 'Template_Reconstruction_mean_shape.stl'
        writeTriangularSurfaceMesh(vertices_mean, faces, mean_name)

        # first generated mean shape is remeshed to desiered vertex number
        # if j == 1:
        print('-----------------------------------')
        print('Remeshing the mean FIRST mean shape')
        mean_name = isotropic_remeshing_target(mean_name, target_nr, 'Template_', forced=True)
        print('-----------------------------------')

        # mean shape : STL2VTK
        stl_file = 'Template_Reconstruction_mean_shape.stl'
        filename = stl_file.split('.')[0]
        mesh = pv.read(stl_file)
        mesh.save(filename + ".vtk", binary=False)

        template_path = os.path.join(data_base, 'Template_Reconstruction_mean_shape.vtk')  # template path
        # print('Template path', template_path)


    # single template
    template_specifications = {
        object_name: {'deformable_object_type': 'surfacemesh',
                      'kernel_type': 'keops', 'kernel_width': kernel_width, 'noise_std': 0.1,
                      'filename': template_path,
                      'attachment_type': 'varifold'}
    }


    # Scipy method
    '''
    estimator_options={'optimization_method_type': 'ScipyLBFGS', 'max_iterations': 90,
                        'convergence_tolerance':10**(-8), 'max_line_search_iterations':20,
                        'freeze-template': "Off",
                        'freeze-control-points':"Off",
                        'callback': estimator_callback}
    '''
    # GradientAscent method - max it ~ 90
    estimator_options = {'optimization_method_type': 'GradientAscent', 'initial_step_size': 1.,
                         'max_iterations': def_it, 'max_line_search_iterations': 20,
                         'convergence_tolerance': 10 ** (tol_exp), 'callback': estimator_callback}


    # create the output dir if doesn't exists and delete the previous files
    filelist = glob.glob(os.path.join(data_out, "*"))
    for f in filelist:
        os.remove(f)
    isExist = os.path.exists(data_out)
    os.makedirs(data_out, exist_ok=isExist)

    # instantiate a Deformetrica object
    deformetrica = dfca.Deformetrica(output_dir=data_out, verbosity='INFO')

    # perform a deterministic atlas estimation
    model = deformetrica.estimate_deterministic_atlas(template_specifications, dataset_specifications,
                                                      estimator_options=estimator_options,
                                                      model_options={'deformation_kernel_type': 'keops',
                                                                     'deformation_kernel_width': kernel_width, 'dtype': 'float32'})
    # check memory usage
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)  # in bytes

    # VTK2STL
    output_files = glob.glob(data_out + "*.vtk")

    for output_file in output_files:
        filename = output_file.split('.')[0]
        mesh2 = pv.read(output_file)
        mesh2.save(filename + ".stl", binary=False)

    t[j+1] = time.time()


###############################################################################
#                              print logs
###############################################################################
print("------------------------------")
print("print log files and statistics")
print("------------------------------")

# print times
for i in range(nr_it):
    print(f'\nIteration nr {i} - time :  ', round((t[i+1] - t[i])/60,3)  , '   min')
print('\n')
print(f'Total time of computation for {nr_it} iterations : ', round((t[-1] - t[0])/60,3), '   min')

# write times in log file
os.chdir(data_path)
time_file = 'time_info.log'
with open(time_file, 'w') as f:
    for i in range(nr_it):
        f.write(f'Iteration nr {i} - time :  ' +  str(round((t[i+1] - t[i]) / 60, 3)) +  '   min')
        f.write('\n')
    f.write('\n')
    f.write(f'Total time of computation for {nr_it} iterations in minutes:\n')
    f.write(str(round((t[-1] - t[0])/60,3)))


# write kernel width
kernel_file = 'kernel_width.log'
with open(kernel_file, 'w') as f:
    f.write("Kernel width used:\n")
    f.write("In percent of diagonal " + str(round(per,3)) + "\n")
    f.write(str(kernel_width))

# delete unwanted files
dir = os.path.abspath('/home/user/Documents/data_deformetrica/')
remove_files_including_string(dir, '__flow__')


###############################################################################
#                          plot log-likelihood
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
# config InlineBackend.figure_format = 'retina'


# print available saved status keys
print("-----------------------")
print("Plotting log-likelihood")
print("-----------------------")
# print(iteration_status_dictionaries[-1].keys())
# print(iteration_status_dictionaries[-1]['gradient'].keys())

max_iterations = len(iteration_status_dictionaries) # for the len of x
x = np.arange(1, max_iterations+1)
y =  [abs(it_data['current_log_likelihood']) for it_data in iteration_status_dictionaries]

# plot log-likelihood
f = plt.figure()
plt.plot(x, y, label='log_likelihood')
# plt.plot(x, [it_data['current_attachment'] for it_data in iteration_status_dictionaries], label='attachment')
# plt.plot(x, [it_data['current_regularity'] for it_data in iteration_status_dictionaries], label='regularity')

plt.yscale('log')

plt.xticks(x)

# plt.style.use('default')
plt.legend()
# plt.show()
log_likeli = os.path.join(data_path, "log_likelihood.png")
plt.savefig(log_likeli)

# write log likelihood in extra log file as table
log_likeli_file = 'likelihood.log'
with open(log_likeli_file, 'w') as f:
    for i in range(len(y)):
        # x value is only range(len(y))
        f.write(str(y[i]) + '\n')
