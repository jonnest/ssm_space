################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
import numpy as np
import pyvista as pv
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib as mpl

import seaborn as sns


def display_correlation(df, save_path, corr_method="pearson"):
    '''
    Display the correlation in a heatmap matrix.
    Default is the pearson correlation, e.g. also possible to use the spearman
    correlation.

    Parameters
    ----------
    df : pd.Dataframe
        Pandas dataframe containing multiple datasets to analyze within correlations.
    save_path : string
        File name with absolute path to save the plot.
    corr_method : string
        Correlation coefficient method, e.g. pearson, spearman, kendall etc.
        Default is pearson.

    Returns
    -------
    None.

    '''
    r = df.corr(method=corr_method)
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(r, vmin=-1, 
                      vmax=1, annot=True, cmap="RdBu")
    plt.title(f"{corr_method} Correlation")
    
    plt.savefig(save_path, dpi=300)
    
    return(r)


def plot17segments(z, save_file, title, min_val=0, max_val=0):
    '''
    modified plot17segments function - OPTIONAL scaling of the color bar 
    set lowest and highest boundary of scale
    min/max val need to be lower/higher than the min/max of z to set limits
    
    Parameters
    ----------
    z : (m,)
        1D np array of size m, z value for each segment.
    save_file : string
        path with file name to save polar plot.

    Returns
    -------
    None.

    '''

    '''
    Polar plot with color bar
    https://stackoverflow.com/questions/16264837/how-does-one-add-a-colorbar-to-a-polar-plot-rose-diagram
    '''
    # add min_val and max_val range values to scalar array   
    z = np.append(z, [min_val, max_val])     
    
    # starting angles of arc: segments sorted as basal: 1-6, mid: 7-12, apical: 13-16, apex: 17
    # add 2 plot points (minimal size - not visible) for scaling of the polar plot
    start_arc = np.array([ 2 *np.pi/6, 4 *np.pi/6, 6 *np.pi/6, 8 *np.pi/6, 10 *np.pi/6, 0 *np.pi/6,
                        2 *np.pi/6, 4 *np.pi/6, 6 *np.pi/6, 8 *np.pi/6, 10 *np.pi/6, 0 *np.pi/6,
                        1 *np.pi/4,  3 *np.pi/4,  5 *np.pi/4,  7 *np.pi/4, 
                        0,
                        0, 0]) 
    # center of arc segments for text position
    ctr_arc = np.array([ 3 *np.pi/6, 5 *np.pi/6, 7 *np.pi/6, 9 *np.pi/6, 11 *np.pi/6, 1 *np.pi/6,
                         3 *np.pi/6, 5 *np.pi/6, 7 *np.pi/6, 9 *np.pi/6, 11 *np.pi/6, 1 *np.pi/6,
                        2 *np.pi/4,  4 *np.pi/4,  6 *np.pi/4,  0 *np.pi/4, 
                        0,
                        0, 0])
    # bottom of segment
    bottom_plot = np.array([15,15,15,15,15,15,10,10,10,10,10,10,5,5,5,5,0,0,0])
    # height of segment
    height_plot = 5*np.ones(17)
    height_plot = np.append(height_plot, [0.0001, 0.0001])
    # arc width 
    width_plot = np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, 
                      np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, 
                      np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi*2, 
                      0.0001, 0.0001])
    
    # initalize the polar bar plot for correct visualization - important 
    fig = plt.figure() #figsize=(10,8))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(0, 10, width=0.3, bottom=0.0, color='Red', alpha=0.5)
    plt.close()
    
    # for evaluation use font 54
    font = {#'family' : 'normal',
            #'weight' : 'normal',
            'size'   : 54}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    plt.rc('font', family='mathptmx')

    plt.rcParams['figure.figsize'] = [10, 8]
    fig = plt.figure()
    # ax = fig.add_subplot(plot_config, polar=True)
    # ax = fig.add_subplot(111, polar=True)
    
    ax = fig.add_subplot(111, projection='polar')
    
    x = start_arc
    y = height_plot
    cmap = plt.get_cmap('RdBu') # cool, jet, colormap-diverging: coolwarm, seismic, RdBu (use for mode excitation)
    coll = colored_bar(x, y, bottom_plot, width_plot, z, ax=ax, cmap=cmap)
    
    norm = mpl.colors.Normalize(vmin=5, vmax=10)
    
    fig.colorbar(coll, label=title, norm=norm ) ,#  ticks=ax_lim) #boundaries=ax_lim)
    #ax.set_yticks([0.5, 1.0])
    #ax.set_rlabel_position(-22.5)
    #ax.grid(True)
    ax.set_ylim(0,20)
    ax.set_xlim(0,2*np.pi)

    # weight: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight'
    for i in range(16): 
        ax.text( ctr_arc[i], bottom_plot[i]+2.5 , f'{i+1}' , color='black',
                weight='normal', horizontalalignment='center', verticalalignment='center')
    ax.text( ctr_arc[16], bottom_plot[16] , "17" , color='black',
            weight='normal', horizontalalignment='center', verticalalignment='center')
     
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_file) #, dpi=400)
    
    return



def colored_bar(left, height, bottom,  width, z=None, ax=None, **kwargs):
    '''
    Polar plot with colar bar

    Parameters
    ----------
    left : (k,) numpy array
        circumferential beginning of patch measured in radians.
    height : (k,) numpy array
        radial height of patch.
    bottom : (k,) numpy array
        .
    width : (k,) numpy array
        circumferential width of patch.
    z : (k,) numpy array, optional
        scalar value plotted on patch. The default is None.
    ax : ax, optional
        plot polar values on ax figure. The default is None.
    **kwargs : e.g. cmap
        colormap.

    Returns
    -------
    coll : TYPE
        DESCRIPTION.

    '''
    if ax is None:
        ax = plt.gca()
    
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    
    rects = []
    for x, y, w, h in zip(left, bottom, width, height):
        rects.append(Rectangle((x,y), w, h))
    coll = PatchCollection(rects, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale()
    
    return coll


def plotShapeFromVertice(vertices,faces,save_file,view=None):
    '''
    Plot shape via vertices and faces

    Parameters
    ----------
    vertices : (N, dim) numpy nd array
        coordinates of data points of shape.
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.
    save_file : string
        path with file name to save shape plot.
    view : string, optional
        set specific 'ventricle' view. The default is None.

    Returns
    -------
    None.

    '''
    faces_pv = np.concatenate((3*np.ones(faces.shape[0])[:,np.newaxis],faces),axis=1)
    faces_pv = np.array(faces_pv.flatten(), dtype=int)
    mesh = pv.PolyData(vertices,faces_pv)

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    pv.global_theme.show_edges = True
    pv.global_theme.font.size = 24

    # my_theme = pv.themes.DocumentTheme()
    # my_theme.cmap = 'jet'
    # my_theme.show_edges = False
    # pv.global_theme.load_theme(my_theme)

    pl = pv.Plotter()
    pl.add_mesh(mesh)
    if view=='ventricle':
        pl.view_xz()
        pl.show_grid()
        pl.camera.roll += 180
        pl.camera.zoom(1)
    else:
        pl.view_xy()
        pl.show_grid()
    pl.show(screenshot=save_file, window_size=[1250,1200]) 
    
    return 


def plotShapeFromFile(msh_file, *save_file):
    '''
    Plot shape from mesh file, e.g. .stl file

    Parameters
    ----------
    msh_file : string, .stl file
        path and name of file.
    save_file : string
        path with file name to save shape plot.

    Returns
    -------
    None.

    '''

    mesh = pv.read(msh_file)

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    pv.global_theme.show_edges = True
    pv.global_theme.font.size = 24
    
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.view_xz()
    pl.camera.roll += 10 #180
    pl.camera.azimuth = -20
    pl.camera.elevation = 30
    
    pl.camera.zoom(1)
    # pl.show_grid()
    
    if save_file == ():
        pl.show()
    else:
        pl.show(screenshot=save_file[0], window_size=[1250,1200])
    
    #mesh.save('ventricle.ply') # .ply .stl .vtk .vtp
    
    return 


def plotScalarOnShape(msh_file,scalar,scalar_name,save_file,*arg):
    '''
    Plots scalar value on surface of geometry
    
    Parameters
    ----------
    msh_file : '<name>.stl'
        plotted geometry file.
    scalar : (N, ) numpy nd array
        scalar value on vertices.
    scalar_name : string
        name of scalar.
    save_file : string
        file name with ending ['.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff']

    Returns
    -------
    None.

    '''
    mesh = pv.read(msh_file)
    #sargs = dict(interactive=True)
    sargs = dict(height=0.7, vertical=True, 
                 position_x=0.85, position_y=0.05,
                 title=scalar_name,
                 title_font_size=20,
                 label_font_size=16,
                 )

    #annotations = {
    #    0.75*np.max(dist_rel): "High",
    #    0.1*np.max(dist_rel): "Low",
    #}

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    
    my_theme = pv.themes.DocumentTheme()
    my_theme.cmap = 'jet'
    my_theme.show_edges = False
    pv.global_theme.load_theme(my_theme)


    pl = pv.Plotter()
    mesh["scalars"] = scalar
    #mesh.plot(scalar_bar_args=sargs)
    pl.add_mesh(mesh, show_scalar_bar = False)
    pl.add_scalar_bar(height=0.7, vertical=True, 
                 position_x=0.84, position_y=0.05,
                 title=scalar_name,
                 title_font_size=40,
                 label_font_size=40)
    
    pl.view_xz()
    pl.camera.roll += 180       # turn around x-axis
    pl.camera.azimuth += 20      # turn around z-axis
    pl.camera.elevation += 30   # turn around y-axis
    if arg!=():
        pl.add_title(title=arg[0])
        
    pl.show(screenshot=save_file, window_size=[1250,1200]) 
    
              #, annotations=annotations)
              
    #mesh.save('ventricle.ply') # .ply .stl .vtk .vtp
    
    return 



def plotSegmentsOnShape(msh_file,scalar,scalar_name,save_file,*arg):
    '''
    Plots scalar value on surface of geometry
    
    Parameters
    ----------
    msh_file : '<name>.stl'
        plotted geometry file.
    scalar : (N, ) numpy nd array
        scalar value on vertices.
    scalar_name : string
        name of scalar.
    save_file : string
        file name with ending ['.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff']

    Returns
    -------
    None.

    '''
    mesh = pv.read(msh_file)
    #sargs = dict(interactive=True)
    sargs = dict(height=0.7, vertical=True, 
                 position_x=0.85, position_y=0.05,
                 title=scalar_name,
                 title_font_size=20,
                 label_font_size=16,
                 )

    #annotations = {
    #    0.75*np.max(dist_rel): "High",
    #    0.1*np.max(dist_rel): "Low",
    #}

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'

    col_list = ['lavender', 'Blue', 'Orange', 'navy', 'Red', 'Yellow', 
                 'Orange', 'Purple', 'Red', 'gold', 'limegreen', 'Blue',
                'Yellow', 'magenta', 'silver', 'Lightblue', 'Orange', 'aqua']
    
    my_theme = pv.themes.DocumentTheme()
    my_theme.cmap = col_list #'jet'
    my_theme.show_edges = False
    pv.global_theme.load_theme(my_theme)


    pl = pv.Plotter()
    mesh["scalars"] = scalar
    #mesh.plot(scalar_bar_args=sargs)
    pl.add_mesh(mesh, show_scalar_bar = False)
    pl.add_scalar_bar(height=0.7, vertical=True, 
                 position_x=0.84, position_y=0.05,
                 title=scalar_name,
                 title_font_size=40,
                 label_font_size=40,
                 #n_colors=18,
                 #n_labels=18
                 )
    
    pl.view_xz()
    pl.camera.roll += 180       # turn around x-axis
    pl.camera.azimuth += 20      # turn around z-axis
    pl.camera.elevation += 30   # turn around y-axis
    if arg!=():
        pl.add_title(title=arg[0])
        
    pl.show(screenshot=save_file, window_size=[1250,1200]) 
    
              #, annotations=annotations)
              
    #mesh.save('ventricle.ply') # .ply .stl .vtk .vtp
    
    return 