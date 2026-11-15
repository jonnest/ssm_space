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
import os
import re
import warnings

def current_path():
    '''
    Print current path
    '''
    print("Current working directory")
    print(os.getcwd())
    print()


def reshapeFacesToPolyData(faces):
    '''
    Reshape the faces to fit the format of pyvista polydata

    Parameters
    ----------
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.

    Returns
    -------
    faces_pv : (M, dim+1) numpy nd array
        definition of triangles via vertex indices, with extra integer.
    '''
    faces_pv = np.concatenate((3*np.ones(faces.shape[0])[:,np.newaxis],faces),axis=1)
    
    return faces_pv




def readTriangularSurfaceMesh(path):
    '''
    read triangluar mesh from path

    Parameters
    ----------
    path : string
        path and name of mesh file.

    Returns
    -------
    vertices : (N, dim) numpy nd array
        coordinates of data points of shape.
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.
    '''
    mesh = pv.read(path)
    vertices,faces = mesh.points,(mesh.faces.reshape(int(mesh.faces.shape[0]/4),4)[:,1:])

    return vertices,faces


def writeTriangularSurfaceMesh(vertices, faces, path):
    '''
    Write triangular mesh to .stl file from vertices and faces

    Parameters
    ----------
    vertices : (N, dim) numpy nd array
        coordinates of data points of shape.
    faces : (M, dim) numpy nd array
        definition of triangles via vertex indices.
    path : string
        path and name of mesh file.

    Returns
    -------
    '''

    faces = np.concatenate((3*np.ones(faces.shape[0])[:,np.newaxis],faces),axis=1)
    mesh = pv.PolyData(vertices, faces.flatten().astype(int))
    mesh.save(path)


def remove_files(w_dir, file_ending):
    '''
    remove files in work directory with specific extension

    Parameters
    ----------
    w_dir : string
        directory to walk through to delete files
    file_ending : string
        files with specific extension/ending which are deleted

    Returns
    -------
    '''
    os.chdir(w_dir)
    cases = os.listdir()

    for case in cases:
        print(case)
        os.chdir(case)
        current_path()
        files = os.listdir()

        for file in files:
            if os.path.isfile:        
                if file.endswith(file_ending):
                    os.remove(file)
        os.chdir('..')
        

def remove_files_including_string(dir, string):
    '''
    Remove all files in directory and subfolders containing specific 'string' in name

    Parameters
    ----------
    dir : string
        directory to walk through and delete files
    string : string
        specific string to search for in file names

    Returns
    -------
    '''
    i = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            if string in name:
                os.remove(os.path.join(root, name))
                i += 1

    print("-----------------------")
    print(i, ' files deleted')
    print("-----------------------")


def select_files_with_extension(input_list, ext):
    '''
    select sublist from list with all items with specific extension 'ext'

    Parameters
    ----------
    input_list : (N,) list
        list with multiple file types
    ext : string
        extension to choose

    Returns
    -------
    output_list : (M,) list
        list with file type of 'ext' only
    '''
    output_list = []
    for file in input_list:
        if file[-len(ext):] == ext:
            output_list.append(file)
    return output_list


def filtering_func(pair):

    # filter dictionary by value
    key, value = pair
    if value == 1:
        return True  # keep pair in the filtered dictionary
    else:
        return False  # filter pair out of the dictionary


def get_number_of_items(metric_names):
    
    # number of metrics to use, count items equal 1 in dictionary
    nr_props = 0
    for count, key in enumerate(metric_names.keys()):
        if metric_names[key] == 1:
            nr_props += 1
    
    return nr_props


def read_surface(filename):
    fileType = get_filetype(str(filename))
    stl = '.stl'
    obj = '.obj'

    if fileType.lower() == stl:

        try:
            vertices, normals, faces, v1, v2, v3 = read_ascii(filename)
            print("File was a STL ascii file. Surface has ", vertices.shape[0],
                  " vertices and ", faces.shape[0], " faces.")
        except UnicodeDecodeError:
            vertices, normals, faces, v1, v2, v3 = read_binary(filename)
            print("File was a STL binary file. Surface has ", vertices.shape[0],
                  " vertices and ", faces.shape[0], " faces.")


    elif fileType.lower() == obj:
        vertices, normals, faces, v1, v2, v3 = readOBJ(filename)
        print("File is an OBJ file. Surface has ", vertices.shape[0],
              " vertices and ", faces.shape[0], " faces.")

    else:
        raise TypeError("Unknown file type for reading surfaces. Must be '.stl' or '.obj'!")
        # print("Unknown file type for reading surfaces.")

    return vertices, normals, faces, v1, v2, v3


def replaceAll(file, searchExp, replaceExp):
    '''
    Replace lines with specific strings.

    Parameters
    ----------
    file : string
        Path to target txt file.
    searchExp : string
        In every line this expression occures, replace with replaceExp.
    replaceExp : string
        Replacement string for each line with occurence.

    Returns
    -------
    None.

    '''
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(line,replaceExp)
        sys.stdout.write(line)

    return


def read_binary(filename):
    warnings.warn('Warning: Binary files can only be processed in little entian format!')
    with open(str(filename), 'rb') as f:
        header = f.read(80)
        nn = f.read(4)
        ntri = unpack('i', nn)[0]
        record_dtype = np.dtype([
            ('normals', np.float32, (3,)),
            ('vertex1', np.float32, (3,)),
            ('vertex2', np.float32, (3,)),
            ('vertex3', np.float32, (3,)),
            ('atttr', '<i2', (1,))
        ])
        data = np.fromfile(f, dtype=record_dtype, count=ntri)

    normals = data['normals']
    v1 = data['vertex1']
    v2 = data['vertex2']
    v3 = data['vertex3']

    nFaces = len(v1)
    vertices = np.zeros((3 * nFaces, 3))
    vertices[::3, :] = v1
    vertices[1::3, :] = v2
    vertices[2::3, :] = v3

    faces = np.reshape(np.arange(3 * nFaces), (nFaces, 3))
    # print(faces)
    # faces = np.array(faces, dtype=np.int64)
    vertices, faces = slimstl(vertices, faces)
    vertices = np.array(vertices, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)

    return vertices, normals, faces, v1, v2, v3


def read_ascii(filename, slim_vertices=True):
    ''' Reads Ascii STL files and returns the vertices, normals and v1, v2, v3
        which form a triangle. Return of "faces" with indices of vertices to
        form a triangle is possible. '''
    normals = []
    vertices = []
    faces = []
    v1 = []
    v2 = []
    v3 = []

    nFaces = 0
    nVertices = 0
    # Patterns to read Ascii file
    pat_patch = re.compile(r'\b\"\w*\"\b')
    pat_facet_normal = re.compile(r'\bfacet normal\b')
    pat_vertex = re.compile(r'\bvertex\b')
    pat_number = re.compile(r'-?\d+\.?\d*')  # int or decimal number
    pat_number_scientific = re.compile(r'-?\d+\.?\d*(?:e-?\d+)?')
    pat_endfacet = re.compile(r'\bendfacet\b')

    # pat_number_exp = re.compile(r'\b[-+]?[0-9]*[.,]?[0-9]+([eE][-+]?[0-9]+)?\b')
    # pat_number_exp = re.compile(r'\b[-+]?[0-9]*[.,]?[0-9]?+([eE][-+]?[0-9]+)?\b')
    with open(str(filename), 'r') as f:

        for line in f:
            if pat_patch.search(line) != None:
                stlname = pat_patch.findall(line)

            if pat_facet_normal.search(line) != None:
                normals.append(pat_number_scientific.findall(line))

            if pat_vertex.search(line) != None:
                nVertices += 1
                vert_buffer = pat_number_scientific.findall(line)
                vertices.append(vert_buffer)

            if pat_endfacet.search(line) != None:
                nFaces += 1
                last_index = len(vertices) - 1
                faces.append([last_index - 2, last_index - 1, last_index])

    # print(np.array(vertices))
    vertices = np.array(vertices, dtype=np.float64)

    faces = np.array(faces)
    normals = np.array(normals, dtype=np.float64)

    if nVertices == 3 * nFaces & slim_vertices == True:
        vertices, faces = slimstl(vertices, faces)

    nfaces = faces.shape[0]
    v1 = np.zeros([nfaces, 3])
    v2 = np.zeros([nfaces, 3])
    v3 = np.zeros([nfaces, 3])

    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    return vertices, normals, faces, v1, v2, v3


def readOBJ(filename):
    normals = []
    vertices = []
    faces = []
    v1 = []
    v2 = []
    v3 = []

    nFaces = 0
    nVertices = 0

    pat_face_normal = re.compile(r'\bvn\b')
    pat_vertex = re.compile(r'\bv\b')
    pat_face = re.compile(r'\bf\b')
    pat_number_v = re.compile(r'-?\d+.\d+')
    pat_number = re.compile(r'-?\d+\.\d+(?:e-?\d+)?')  # allowing scientific format for numbers
    pat_number_scientific = re.compile(r'-?\d+\.?\d*(?:e-?\d+)?')
    pat_int = re.compile(r'\d+')

    with open(str(filename), 'rt') as f:

        for line in f:

            if pat_vertex.search(line) != None:
                nVertices += 1
                vertices.append(pat_number_scientific.findall(line))
                # print(vertices)
            if pat_face_normal.search(line) != None:
                normals.append(pat_number_scientific.findall(line))
                # print(normals)
            if pat_face.search(line) != None:
                nFaces += 1
                faces.append(pat_int.findall(line))
    faces = np.array(faces, dtype=np.int64)
    faces = np.unique(faces - 1, axis=1)
    vertices = np.array(vertices, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)

    normals = get_face_normals(vertices, faces)

    nfaces = faces.shape[0]
    v1 = np.zeros([nfaces, 3])
    v2 = np.zeros([nfaces, 3])
    v3 = np.zeros([nfaces, 3])

    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    if nVertices == 3 * nFaces:
        vertices, faces = slimstl(vertices, faces)

    return vertices, normals, faces, v1, v2, v3


def writeAscii(filename, vertices, normals, faces):
    '''
    Write geometry as ascii file.
    
    Parameters
    ----------
    filename : string
        absolute path to save the .obj file.
    vertices : (N, dim) numpy nd array
        coordinates of the object.
    normals : (M, dim) numpy nd array
        normal vector of each face.
    faces : (M, dim) numpy nd array
        faces of the mesh defined by vertex id's

    Returns
    -------

    '''
    n = len(faces)
    solid = filename
    with open(filename, 'w') as f:
        f.write("solid %s\n\n" % solid)
        for i in range(n):
            f.write("  facet normal %r %r %r\n" % tuple(normals[i, :]))
            f.write("    outer loop\n")
            f.write("      vertex %r %r %r\n" % tuple(vertices[faces[i, 0], :]))
            f.write("      vertex %r %r %r\n" % tuple(vertices[faces[i, 1], :]))
            f.write("      vertex %r %r %r\n" % tuple(vertices[faces[i, 2], :]))
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid %s\n" % solid)


def writeOBJ(filename, vertices, faces, *args):
    '''
    Write vertices to .obj file.

    Parameters
    ----------
    filename : string
        absolute path to save the .obj file.
    vertices : (N, dim) numpy nd array
        coordinates of the object.
    faces : (M, dim) numpy nd array
        faces of the mesh.
    
    Returns
    -------
    None.
    
    '''       
    
    nv = len(vertices)
    nf = len(faces)
    with open(filename, 'w') as f:
        if args != None:
            f.write(f"#{args}\n")
        # loop over vertices
        for i in range(nv):
            f.write("v %r %r %r\n" % tuple(vertices[i,:]) )
        f.write(f"# {nv} vertices, 0 vertices normals\n")
        f.write("\n")
        # loop over faces
        for i in range(nf):
            f.write("f %r %r %r\n" % tuple(faces[i, :]+1) )
        f.write(f"# {nf} faces, 0 coords texture\n")
        f.write("\n")
        f.write("# End of File")
    return

def get_filetype(filename):
    name, fileType = os.path.splitext(str(filename))
    return fileType


def slimstl(oldVertices, oldFaces):
    newVerts, indices, inv_indices = np.unique(oldVertices, return_index=True, return_inverse=True, axis=0)
    newFaces = inv_indices[oldFaces]
    print('dublicates in vertice list are being removed...')
    print('old number vertices: ', len(oldVertices))
    print('new number vertices: ', len(newVerts))
    print('old number of faces: ', len(oldFaces))
    print('new number of faces: ', len(newFaces), '... should not change!')
    return newVerts, newFaces


def get_face_normals(vertices, faces):
    '''
    Calculates face normals and ensures right order in indices. This
    is not necessarily the case with data read from binary files

    Parameters
    ----------
    vertices : (N, dim) numpy nd array
        coordinates of the object.
    faces : (M, dim) numpy nd array
        faces of the mesh.

    Returns
    -------
    faceNormals : (N, dim)
        matrix of faces normals, normals for each face.
    '''
    v1 = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    normals = np.cross(v1, v2, axisa=1, axisb=1)
    norm = np.linalg.norm(normals, axis=1)
    fnormals = [normals[:, i] / norm[:] for i in range(3)]
    faceNormals = np.array(fnormals, dtype=np.float64).T

    return faceNormals  # error in some normals suspected after read-in

