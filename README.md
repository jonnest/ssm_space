# Statistical Shape Modeling Library


## Description
The Statistical Shape Modeling Library is a repository for the shape analysis of 
anatomical structures. The functionalities of the library are: 
- read mesh files from specific directory structures
- manipulate, align, etc. meshes and write new files
- general processing and analysis of shape features/metrics
- construct a SSM based on the linear or non-linear method
- process single or multiple time states of patient-specific geometries in the SSM
- generation of synthetic shapes based on random sampling of the eigenvalues 
- generation of synthetic shapes based on a Copula using the shape parameters of the SSM 
- optional processing of the demographic data of patients
- statistical analysis and comparison of the synthetic and original data sets
    based on anatomical features
- analysis of correlation of anatomical features
- filtering of the data, specifically the synthetic cohort
- plot functions and visualization tools for python framework



## Remarks
- when running on Linux (e.g. for nonlinear SSM usage) issues with the header 
names in the TomTec file occure
- nonlinear SSM only with separate states possible yet



## Roadmap
Future implementation of:
- exact calculation of the 3d orifice area (Juliana's code) is implemented, 
yet some shapes are causing errors : implement skip feature with NaN values for 
the metrics and use e.g. np.nanmean() etc. later on
- Deformetrica workflow included in the project structure
- non-linear SSM: hierarchical model for the non-linear SSM based on multiple states 
- general repository for all projects using SSM analysis
- extent the multiple SSM to more than two states and check transition from diastolic to systolic 
state for corresponding synthetic mitral valves


## Structure of the Git
### Directories
- **ssm** : ssm functionalities and scripts
  * **ssm_general**
  * **ssm_mv_tomtec**
    * infile.py : set all paths, parameters and settings for the analysis
    * execute main.py for analysis of a mitral cohort obtained via TomTec
  * **ssm_ventricle**
- **tools** : various functions
- **test** : run test routines to check functionality of the Git
- **deformetrica** : tools for generation of nodal correspondence
- **docs** : documentation
  * _build/html/index.html : file to open for Sphinx documentation of the Git
- **morphomatics** : Morphomatics 3.0 library for the non-linear SSM, version 3.0
contains the hierarchical feature to implement multiple states
- **misc** : Miscellaneous files, e.g. YML file for Python environments

### Packages and programs 
Usage and its requirements for environments:
- Gerneral geometry analysis : Python>3.8
- **Morphomatics** : Python>3.8, full functionality only on Linux OS
- **Deformetrica** : Clean install with Python=3.8, only on Linux OS, usage of PyCharm
- **PyMeshLab** : not compatible with Spyder on Windows, either use Jupyter Lab or
              PyCharm; more robust on Linux OS

## Project: SSM_MV_TOMTEC
### Structure of the SSM_MV_TOMTEC workspace 
Data used in the project needs to follow a specific data structure.
First, the code requires a given workspace "workspace1" directory containing all subfolders
for a specific analysis.
- /workspace1
  * CasesOverview_pre.xlsx (REQUIRED)
  * Demographic_Data_anonymized.xlsx (REQUIRED)
  * TomTec-Results.xlsx (REQUIRED)
  * faces_ant.txt (REQUIRED)
  * faces_post.txt (REQUIRED)
  * /infile
      * infile.txt
      * save_parameter_file.txt
  * /metrics
  * /OBJ (REQUIRED)
    * /case1
      * /state1
      * /state2
    * /case2
       * ...
  * /plots
  * /ssm
  * /stats
  * /synthetic
    * /state1
      * shape0001.obj
      * shape0002.obj
      * ...
    * /state2
      * shape0001.obj
      * shape0002.obj
      * ...

For a new analysis on the mitral data set used in the SSM_MV_TOMTEC, a default
workspace is provided. As seen above, the default workspace contains all
folders and files marked with 'required'. 

### Usage and application
- **Alignment**
  * Landmark : aligning local coordinate systems defined via the commissure points and the center of mass
  * Iterative closet point (ICP) algorithm : not fully implemented yet, but function available in tools.pointclouds.py
- **Statistical Shape Modeling**
  * Method
    * linear
    * non-linear (Morphomatics : use student P1 Thinkpad, PyCharm: "morphomatics" environment)
  * Structure
    * single state (each time-specific state processed in separate SSM)
    * multiple state (all time-specific states processed in one large SSM)
  * Reconstruction
    * randomized sampling of the shape parameter with normal distribution (mean=0, std=eigenvalue_i)
    * Copula of the original shape parameters calculated via linear equation system using the modes (only linear SSM)
      * possible to add demographic data
- **Metrics / Anatomical Features**

| Variable names      |      Description                                                             | TomTec Availability |
| ------------------- | ---------------------------------------------------------------------------- | ------------------- |
| A_annulus           |  projected area of the annulus boundary curve                                | yes                 |
| circ_annulus        |  circumference of the 3d annulus boundary curve                              | yes                 |
| ci_annulus          |  circularity index of the annulus boundary curve                             | yes                 |
| h_annulus           |  height (z-direction in aligned mode) of annulus boundary curve              | yes                 |
| ratio_annulus       |  ratio of x/y dimension of aligned mode of annulus boundary curve            |                     |
| circ_orifice        |  circumference of the 3d orifice boundary curve                              |                     |
| ci_orifice          |  circularity index of the orifice boundary curve                             |                     |
| h_orifice           |  height (z-direction in aligned mode) of orifice boundary curve              |                     |
| ratio_orifice       |  ratio of x/y dimension of aligned mode of orifice boundary curve            |                     |
| len_lf_ant          |  length of anterior leaflet (straight line or path; change in source code)   | yes                 |
| len_lf_post         |  length of posterior leaflet (straight line or path; change in source code)  | yes                 |
| ang_lf_ant          |  angle of anterior leaflet to normal vector of annulus plane                 | yes                 |
| ang_lf_post         |  angle of posterior leaflet to normal vector of annulus plane                | yes                 |
| A_lf_ant            |  3d area of anterior leaflet                                                 | yes                 |
| A_lf_post           |  3d area of posterior leaflet                                                | yes                 |
| A_3d_orifice_approx |  approximation of 3d orifice area (overestimation; fast computation)         |                     |
| A_3d_orifice        |  exact 3d orifice area by Julianas algorithm (ca. 25 secs per geometry)      |                     |
| height_to_width     |  ratio of annulus height to commissure width (cw defined along path of grid) |                     |

- **Plots for Evaluation**
  * Boxplot
  * Histogram
  * Correlation matrix (e.g. Pearson, Spearman)
  * Correlation plots
  * Visualization of the shapes using PyVista
- **Comparison of States** (diastolic and systolic state)
  * Correlation plots
  * Procrustes disparity; difference in shape of the annulus boundary curve
  * Hausdorff distance of the annulus boundary curve
  * Dynamic time warping for the annulus boundary curve
- **Demographic Data**
  * generation of subcohorts according to e.g. sex
- **Filtering of Synthetic Data**
  * self-intersections (via PyMeshLab)
  * filtering according the interquartile range of the real/synthetic data

The documentation with all packages and functions is located here: ``/docs/_build/html/index.html``.

***
## Installation
### Packages and programs
Install Anaconda3 and Spyder for user friendly usage
- [Anaconda3](https://www.anaconda.com/)

Required packages for general usage
```
numpy
pandas
pyvista
scipy
seaborn
matplotlib
os, sys, time
random
pymeshlab
```
- [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/index.html) : not working via Spyder, workaround is executing commands via shall
- [Morphomatics](https://morphomatics.github.io/) only running on Linux, usage of the JAX library
- [Deformetrica](https://www.deformetrica.org/) only running on Linux

Use the YML file from the /misc folder to reinstall all used Python packages.
The following conda command exports the installed packages from the environment:
```
conda env export > environment.yml
```

### How to install Anaconda
Recommended IDE: Spyder or PyCharm
```
conda create -n new_env python=3.8
conda activate new_env
```
See all available environments
```
conda env list
```

For usage of Deformetrica Pycharm is recommended.

Best practice:
- create Anaconda environment
- install packages and modules in environment
- open/create project in IDE (Spyder/PyCharm) and choose Anaconda environment as Python interpreter


### How to install Morphomatics
Prerequisite: Anaconda clean install (for all users)
- download Morphomatics 1.1 directory from Github as zip folder [Github Morphomatics](https://github.com/morphomatics/morphomatics/releases)
- open terminal and go to Morphomatics directory; alternatively open file explorer of Morphomatics folder and open Powershell
- install morphomatics via the setup.py file, but do not allow installation via internet (ideally turn of internet), then install with `python setup.py`
- install missing module pymanopt with `pip install pymanopt==0.2.5`


### Morphomatics (newest version - on linux)
The newest version of Morphomatics does include SSM for hierarchical models.
This can be used to implement the multiple states of the LV (e.g. systolic and diastolic) in the non-linear SSM model.
Using these hierarchical models allows to generate modes on geodesics.

Current status: newest version of Morphomatics not offically available, yet the code includes the unpublished version.
The code requires the JAX library which is straight forward to install on Linux OS, yet on Windows it is complicated (see [Install JAX](https://github.com/google/jax#installation))

In the following is the best practice for Ubuntu LTS 22.04 using Anaconda.
Is running with Python 3.8 and 3.9
For usage with Deformetrica together use Python 3.8
```
conda create -n morphomatics python=3.8
conda activate morphomatics
```
Installation of multiple packages and fixes for plotting with Pyvista
```
pip install pyvista
pip install qtconsole
pip install meshio
conda install -c conda-forge libstdcxx-ng
pip install jax[cpu] pyvista
```

Copy environment to merge Deformetrica environment with Morphomatics environment.
```
conda create --name new_env --clone myenv
```


***
## Documentation with Sphinx
Generate documentation html files with Sphinx. \
Installation
```
sudo apt install python-sphinx -y
pip install sphinx_rtd_theme
```
First argument: output folder, second argument: directory to document
```
sphinx-quickstart
```
Then modify the "conf.py" file. \
Change html_theme = 'alabaster' to 'sphinx_rtd_theme' for clean python representation \
Insert at the top:
```
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
```
Insert at "extensions":
```
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.todo',
        'sphinx.ext.coverage',
]
```
Run the following:
```
sphinx-apidoc -o . ..
```
Modify index.rst file. \
Above 'index and tables' insert modules
```
make html
```

To open the final documentation, go to ``/docs/_build/html/index.html``.


## Usage of Git
Clone the repository
```
git clone https://github.com/jonnest/ssm_space.git
cd ssm_space
```
***

## Authors
Jonathan Nestmann

2026 SSM Space Project
