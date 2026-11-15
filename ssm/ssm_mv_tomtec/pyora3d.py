################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
'''
Computation of exact minimal 3d orifice area from Juliana.

For updates of functions, check "pyora3d" repository in Gitlab.
Note : If functions are updated, decomment all print commands again.

Due to the use of PyMeshLab, the shell workaround is used.

'''

import networkx as nx
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree, distance
from collections import Counter
import trimesh
import pymeshlab as ml
from itertools import product
import os


def get_3D_orifice_area(filename_ant,filename_post,output_folder,output_name,
                        save_mesh = True,accuracy_setting='quick-search',included_neighbor_fraction = 1,
                        quick_search_minimization_rule='area',iter_smooth = 3):
    # print("Prepare meshes and identify orifice(s)..")
    # load meshes
    mesh_ant = pv.read(filename_ant)
    mesh_post = pv.read(filename_post)
    # print("meshes read.")
    # automatically detect mesh intersections and orifice parts with their connectivity
    orifice_parts,mesh_ant_split,mesh_post_split = detect_mesh_intersections_and_orifice_connectivity(mesh_ant,mesh_post)

    # prepare and analyze graph networks
    graph_ant,coord_ant,graph_paths_ant,kdtree_ant = compose_and_analyse_graph_network(mesh_ant_split)
    graph_post,coord_post,graph_paths_post,kdtree_post = compose_and_analyse_graph_network(mesh_post_split)

    # print("Number of identified orifices:", str(len(orifice_parts)))

    # print("Find smallest orifice area(s)...")

    total_orifice_area = []
    for orifice_i in range(len(orifice_parts)):
        if accuracy_setting =='closest_edge_pair':
            edge_combinations = [(0,0)]
            coord_start = list(orifice_parts[orifice_i]['coord_pair'][0])
            coord_end = list(orifice_parts[orifice_i]['coord_pair'][1])
        else:
            coord_start = list(orifice_parts[orifice_i]['coord_1'])
            coord_end = list(orifice_parts[orifice_i]['coord_2'])


            if (orifice_i>0):
                distances = np.linalg.norm(coord_start - orifice_parts[orifice_i]['coord_pair'][1], axis=1)  # Calculate Euclidean distances
                sorted_indices = np.argsort(distances)  # Sort the indices based on distances
                idx_fraction = sorted_indices[:np.round(len(coord_start)*included_neighbor_fraction).astype(int)] # Calculate the index at the xx% mark
                coord_start = [coord_start[index] for index in idx_fraction]

            if (orifice_i<len(orifice_parts)) & (len(orifice_parts)>0) :
                distances = np.linalg.norm(coord_end - orifice_parts[orifice_i]['coord_pair'][0], axis=1)  # Calculate Euclidean distances
                sorted_indices = np.argsort(distances)  # Sort the indices based on distances
                idx_fraction = sorted_indices[:np.round(len(coord_end)*included_neighbor_fraction).astype(int)] # Calculate the index at the xx% mark
                coord_end = [coord_end[index] for index in idx_fraction]


            edge_combinations = list(product(np.arange(len(coord_start)).tolist(), np.arange(len(coord_end)).tolist()))

        if accuracy_setting =='quick-search':
            max_path_smooth = 2

        else:
            max_path_smooth = 100


        orifice_area_opt = 1e10
        orifice_triangulation_opt = []
        all_combis = []

        # coord_start = list(coord_start)
        # coord_end = list(coord_end)
        if len(np.reshape(coord_start,(1,-1))[0])==3:
            coord_start =[coord_start]
        if len(np.reshape(coord_end,(1,-1))[0])==3:
            coord_end =[coord_end]

        for k in range(len(edge_combinations)):

            i = edge_combinations[k][0]
            j = edge_combinations[k][1]

            path_edges_ant = [np.where(np.all(coord_ant == coord_start[i], axis=1))[0][0],np.where(np.all(coord_ant == coord_end[j], axis=1))[0][0]]
            path_edges_post = [np.where(np.all(coord_post == coord_start[i], axis=1))[0][0],np.where(np.all(coord_post == coord_end[j], axis=1))[0][0]]

            orifice_triangulation  = compute_3D_orifice_area(graph_paths_ant,graph_paths_post,kdtree_ant,kdtree_post,coord_ant,coord_post,path_edges_ant,path_edges_post,max_path_smooth,coord_start,coord_end)

            if not orifice_triangulation:
                current_orifice_area = 1e11
            else:
                current_orifice_area = orifice_triangulation.area
                orifice_perimeter = get_perimeter(orifice_triangulation)
                all_combis.append([orifice_perimeter,current_orifice_area,i,j])

            if current_orifice_area<orifice_area_opt:
                orifice_area_opt = current_orifice_area
                orifice_triangulation_opt = orifice_triangulation


        if accuracy_setting =='quick-search':
            if quick_search_minimization_rule == 'area':
                sorted_combis = sorted(all_combis, key=lambda x: x[1])
            elif quick_search_minimization_rule == 'perimeter+area':
                sorted_combis = sorted(all_combis, key=lambda x: x[0]+x[1])

            i = sorted_combis[0][2]
            j = sorted_combis[0][3]
            path_edges_ant = [np.where(np.all(coord_ant == coord_start[i], axis=1))[0][0],np.where(np.all(coord_ant == coord_end[j], axis=1))[0][0]]
            path_edges_post = [np.where(np.all(coord_post == coord_start[i], axis=1))[0][0],np.where(np.all(coord_post == coord_end[j], axis=1))[0][0]]
            max_path_smooth = 100
            orifice_triangulation  = compute_3D_orifice_area(graph_paths_ant,graph_paths_post,kdtree_ant,kdtree_post,coord_ant,coord_post,path_edges_ant,path_edges_post,max_path_smooth,coord_start,coord_end)
            if not orifice_triangulation:
                orifice_triangulation_opt = []
            else:
                orifice_triangulation_opt = orifice_triangulation

        orifice_triangulation_opt.export('stl_temp.stl')
        orifice_triangulation_smooth = remesh_and_smooth_triangulation('stl_temp.stl',iterations =iter_smooth)
        os.remove('stl_temp.stl')
        if save_mesh:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            if len(orifice_parts) > 1:
                orifice_triangulation_smooth.export(os.path.join(output_folder,output_name+'_part_'+str(orifice_i+1)+'.stl'))
            else:
                orifice_triangulation_smooth.export(os.path.join(output_folder,output_name+'.stl'))

        # print('Orifice area part',orifice_i+1,':', np.round(orifice_triangulation_smooth.area, decimals=2), "mm2")

        total_orifice_area.append(orifice_triangulation_smooth.area)

    # print('Total orifice area:', np.round(np.sum(total_orifice_area), decimals=2), "mm2")


    return total_orifice_area

def detect_mesh_intersections_and_orifice_connectivity(mesh_1,mesh_2):
    intersection, mesh_1_split, mesh_2_split = mesh_1.intersection(mesh_2)
    graph_intersection = mesh_to_graph(intersection)
    part_connections, cluster = cluster_intersections(graph_intersection)

    return part_connections,mesh_1_split,mesh_2_split

def compose_and_analyse_graph_network(mesh):
    graph = mesh_to_graph(mesh)
    graph_coords = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    graph_shortest_paths = nx.shortest_path(graph, weight = 'length')
    graph_kdtree = KDTree(graph_coords)

    return graph,graph_coords,graph_shortest_paths,graph_kdtree

def mesh_to_graph(mesh):

    if mesh.faces.size > 0:
        # Get the padded faces
        padded_elements = mesh.faces.reshape(-1)

    elif mesh.lines.size > 0:
        # Get the padded lines
        padded_elements = mesh.lines.reshape(-1)

    else:
        raise ValueError("Mesh has neither faces nor lines.")

    # Create an empty graph
    graph = nx.Graph()

    # Add vertices as nodes to the graph
    for i, vertex in enumerate(mesh.points):
        graph.add_node(i)
        nx.set_node_attributes(graph, {i: tuple(vertex)}, name='pos')

    # Add edges to the graph
    idx = 0
    while idx < len(padded_elements):
        num_points = padded_elements[idx]
        element = padded_elements[idx+1 : idx+1+num_points]
        for i in range(num_points):
            node1 = element[i]
            node2 = element[(i + 1) % num_points]
            pos1 = graph.nodes[node1]['pos']
            pos2 = graph.nodes[node2]['pos']
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            graph.add_edge(node1, node2, length=distance)
        idx += num_points + 1

    return graph

def find_closest_points(coords_set1, coords_set2):
    # Convert coordinates to NumPy arrays
    coords_set1 = np.array(coords_set1)
    coords_set2 = np.array(coords_set2)

    # Compute pairwise distances between points in each set
    pairwise_distances = distance.cdist(coords_set1, coords_set2)

    # Find the indices of the closest points
    min_indices = np.unravel_index(np.argmin(pairwise_distances), pairwise_distances.shape)
    # closest_point1 = coords_set1[min_indices[0]]
    # closest_point2 = coords_set2[min_indices[1]]

    # Compute the distance between the closest points
    distance_closest_points = pairwise_distances[min_indices]

    # Get the positions of the closest points in the original coordinate sets
    index_set1 = min_indices[0]
    index_set2 = min_indices[1]

    return distance_closest_points, index_set1, index_set2

def cluster_intersections(graph_intersection):
    # Perform connected component labeling on the intersection graph
    bins = list(nx.connected_components(graph_intersection))
    bin_temp = [[number, i] for i, bin_set in enumerate(bins) for number in bin_set]
    bin_order = [sublist[1] for sublist in sorted(bin_temp, key=lambda x: x[0])]

    positions = nx.get_node_attributes(graph_intersection, 'pos')

    coord_intersect = np.array(list(positions.values()))

    # Preallocate cluster structure array
    cluster = []
    for _ in range(len(bins)):
        cluster.append({'nodes': [], 'coords': [], 'graph': None})

    # Assign nodes, coordinates, and subgraphs to each cluster
    for i in range(len(bins)):
        cluster[i]['nodes'] = list(bins[i])
        cluster[i]['coords'] = coord_intersect[cluster[i]['nodes'], :]
        cluster[i]['graph'] = graph_intersection.subgraph(cluster[i]['nodes'])

    # Create connected clusters from the intersection graph
    k = 0
    conn_edges = []
    while len(bins) > 1:
        k += 1
        for i in range(len(bins)):
            d_min = 1e5
            for j in range(len(bins)):
                 if i == j:
                     continue
                 else:
                     nodes_i = list(bins[i])
                     nodes_j = list(bins[j])

                     distance_closest_points, index_closest_i, index_closest_j = find_closest_points(coord_intersect[nodes_i,:], coord_intersect[nodes_j,:])
                     if distance_closest_points < d_min:
                        d_min = distance_closest_points
                        nearest_node_i = nodes_i[index_closest_i]
                        nearest_node_j = nodes_j[index_closest_j]
            l_g = len(graph_intersection.edges)
            # an edge is added if it is not already part of the graph
            graph_intersection.add_edge(nearest_node_i, nearest_node_j, length=d_min)
            l_g_new = len(graph_intersection.edges)
            if l_g_new > l_g:
                conn_edges.append([nearest_node_i,nearest_node_j])


        bins = list(nx.connected_components(graph_intersection))
        if k == 100:
            break

         # # Assign clusters to the initial bins using the conn_edges matrix

    conn_clusters = []
    for edge_i in conn_edges:
        conn_clusters.append([bin_order[edge_i[0]], bin_order[edge_i[1]]])


    # Order the connection clusters
    conn_clusters = order_edges(conn_clusters)

    # compose information about cluster connections
    connections = []
    for conn_cluster in conn_clusters:
        coord_1 = cluster[conn_cluster[0]]['coords']
        coord_2 = cluster[conn_cluster[1]]['coords']
        tree = KDTree(coord_1)
        dist, idx = tree.query(coord_2)
        idx_min = np.argmin(dist)
        coord_pair = np.array([coord_1[idx[idx_min]], coord_2[idx_min]])
        # if np.linalg.norm((coord_pair[0] - coord_pair[1]) / coord_pair[1]) > 1e-3:
        connections.append({
            'coord_pair': coord_pair,
            'coord_1': coord_1,
            'coord_2': coord_2
        })

    return connections, cluster

def order_edges(edges):

    # Flatten the list of edges and count the occurrences of each node
    node_counts = Counter(node for edge in edges for node in edge)

    # Find nodes that appear only once
    unique_nodes = [node for node, count in node_counts.items() if count == 1]

    edges = np.array(edges)

    # Put unique node in first column
    if np.sum(np.isin(edges[:, 0], unique_nodes[0])) == 0:
        idx = np.isin(edges[:, 1], unique_nodes[0])
        edges[idx] = np.column_stack((edges[idx, 1], edges[idx, 0]))


    # Put first node in the first row
    start_idx = np.where(edges[:, 1] == unique_nodes[0])[0]
    conn_clusters_initial = edges[start_idx]
    edges = np.delete(edges, start_idx, axis=0)
    edges = np.vstack((conn_clusters_initial, edges))



    num_edges = edges.shape[0]

    # Order nodes
    for i in range(0, num_edges-1):
        node = edges[i, 1]
        idx_node = np.where(edges[:, 0] == node)[0]

        if len(idx_node) == 0:
            idxs = np.where(edges[:, 1] == node)
            idx_node = np.setdiff1d(idxs, i)
            edges[idx_node] = np.column_stack((edges[idx_node, 1], edges[idx_node, 0]))

        edge_temp = edges[idx_node]
        edges = np.delete(edges, idx_node, axis=0)
        edges = np.concatenate((edges[:i+1], edge_temp, edges[i+1:]), axis=0)



    # Print the ordered edges
    # print(edges)

    return edges

def interpolate_points(points, fractions):
    """
    Interpolates new points at fractional positions along a curve defined by a NumPy array of points in 3D.

    Args:
        points (np.ndarray): NumPy array of points in 3D with shape (N, 3), where N is the number of points.
        fractions (list): List of fractions representing the positions along the curve where new points should be interpolated.

    Returns:
        np.ndarray: NumPy array of interpolated points in 3D with shape (M, 3), where M is the number of interpolated points.
    """
    interpolated_points = []
    num_points = len(points)

    for fraction in fractions:
        # Calculate the index of the points bounding the fractional position
        index = int(fraction * (num_points - 1))
        if index == num_points - 1:
            interpolated_points.append(points[index])
            break
        else:
            alpha = fraction * (num_points - 1) - index
            p0 = points[index]
            p1 = points[index + 1]
            interpolated_point = p0 + alpha * (p1 - p0)
            interpolated_points.append(interpolated_point)

    return np.array(interpolated_points)

# def calculate_maximum_edge_length(mesh):
#     # Get vertices and faces from the mesh
#     vertices = mesh.vertices
#     faces = mesh.faces

#     # Calculate edge lengths for each triangle
#     edge_lengths = []
#     for face in faces:
#         p0, p1, p2 = vertices[face]
#         edge_lengths.append(
#             (np.linalg.norm(p1 - p0),
#              np.linalg.norm(p2 - p1),
#              np.linalg.norm(p0 - p2))
#         )

#     # Flatten the edge lengths and find the maximum
#     max_edge_length = np.max(np.concatenate(edge_lengths))

#     return max_edge_length



def unique_rows_preserved_order(input_array):
    if len(input_array.shape)<2:
         _, idx = np.unique(input_array, return_index=True)
         unique_array = input_array[np.sort(idx)]

    else:
         _, idx = np.unique(input_array, return_index=True,axis=0)
         unique_array = input_array[np.sort(idx),:]

    return unique_array

def find_smoothed_path(graph_paths,initial_path,n):

    path_final = []

    if n == 1:
        n_side = len(initial_path) // 2;
    else:
        n_side = (len(initial_path) // (2 * n + 1)) * n + 1

    for i in range(0, n_side - n, n):
        path_final.extend(graph_paths[initial_path[i]][initial_path[i + n]])

    k_1 = np.round(len(initial_path)/2).astype(int)
    k_2 = len(initial_path) - n_side
    if (k_2-n_side)>=n:
        path_final.extend(graph_paths[initial_path[n_side-1]][initial_path[k_1]])
        path_final.extend(graph_paths[initial_path[k_1]][initial_path[k_2]])
    else:
        path_final.extend(graph_paths[initial_path[n_side-1]][initial_path[k_2]])


    for i in range(0, n_side - n, n):
        k = len(initial_path) - n_side + i
        path_final.extend(graph_paths[initial_path[k]][ initial_path[k + n]])

    path_final = unique_rows_preserved_order(np.array(path_final))

    return path_final

def optimize_path(graph_paths_1,kdtree_1,path_edges_1,path_coords_2,n_smooth):

    distances, idx = kdtree_1.query(path_coords_2)
    _, idx_temp = np.unique(idx, return_index=True)
    path_nodes_1 = idx[np.sort(idx_temp)]

    if path_nodes_1[0] != path_edges_1[0]:
        path_nodes_1 = path_nodes_1[path_nodes_1 != path_edges_1[0]]
        path_nodes_1 = np.concatenate(([path_edges_1[0]], path_nodes_1))

    if path_nodes_1[-1] != path_edges_1[1]:
        path_nodes_1 = path_nodes_1[path_nodes_1 != path_edges_1[1]]
        path_nodes_1 = np.concatenate((path_nodes_1, [path_edges_1[1]]))


    path_temp = find_smoothed_path(graph_paths_1,path_nodes_1,1)
    path_nodes_1_final = find_smoothed_path(graph_paths_1,path_temp,n_smooth)

    return path_nodes_1_final

def compute_3D_orifice_area(graph_paths_ant,graph_paths_post,kdtree_ant,kdtree_post,coord_ant,coord_post,path_edges_ant,path_edges_post,max_path_smooth,coord_start,coord_end):

    coord_start = np.array(coord_start)
    coord_end= np.array(coord_end)

    # print(np.linspace(2, max_path_smooth, max_path_smooth-1))
    skip_n = np.linspace(2, max_path_smooth, max_path_smooth-1).astype(int)
    skip_m = np.linspace(2, max_path_smooth, max_path_smooth-1).astype(int)

    initial_path_nodes_post = graph_paths_post[path_edges_post[0]][path_edges_post[1]]
    initial_path_nodes_ant = graph_paths_ant[path_edges_ant[0]][path_edges_ant[1]]

    break_reached = False
    # good_solution_found = False
    orifice_triangulation_opt = []
    for i in range(max_path_smooth-1):
        if break_reached:
            break
        # good_solution = False

        for j in range(max_path_smooth-1):
            if i == 0 and j == 0:
                path_ant = optimize_path(graph_paths_ant,kdtree_ant,path_edges_ant,coord_post[initial_path_nodes_post,:],skip_n[i])
                path_post = optimize_path(graph_paths_post,kdtree_post,path_edges_post,coord_ant[initial_path_nodes_ant,:],skip_m[j])

            else:
                if len(path_ant) < 2 * skip_n[i]:
                    break_reached = True
                    break

                if len(path_post) < 2 * skip_m[j]:
                    break
                else:
                    path_ant = optimize_path(graph_paths_ant,kdtree_ant,path_edges_ant,coord_post[path_post,:],skip_n[i])
                    path_post = optimize_path(graph_paths_post,kdtree_post,path_edges_post,coord_ant[path_ant,:],skip_m[j])

            if len(path_ant) == 0 or len(path_post) == 0:
                orifice_triangulation_opt = []
                orifice_area_opt = 0
                break

            # Remove double entries
            path_ant = unique_rows_preserved_order(path_ant)
            path_post = unique_rows_preserved_order(path_post)

            # Coordinates of the paths
            coord_targets_ant = unique_rows_preserved_order(coord_ant[path_ant])
            coord_targets_post = unique_rows_preserved_order(coord_post[path_post])

            # if any( [len(np.where(np.all(coord_targets_ant[:, None] == coord_start, axis=2).any(axis=1))[0])>2,
            # len(np.where(np.all(coord_targets_ant[:, None] == coord_end, axis=2).any(axis=1))[0])>2,
            # len(np.where(np.all(coord_targets_post[:, None] == coord_start, axis=2).any(axis=1))[0])>2,
            # len(np.where(np.all(coord_targets_post[:, None] == coord_end, axis=2).any(axis=1))[0])>2]):
            #     if good_solution_found:
            #         # print('good solution found')
            #         break
            # else:
            #     good_solution = True


            edge_coords = coord_post[path_edges_post,:]
            coord_targets_ant = np.vstack((edge_coords[0],np.delete(coord_targets_ant, np.all(coord_targets_ant[:, None] == coord_targets_post, axis=2).any(axis=1), axis=0),edge_coords[1]))
            coord_targets_post = np.vstack((edge_coords[0],np.delete(coord_targets_post, np.all(coord_targets_post[:, None] == coord_targets_ant, axis=2).any(axis=1), axis=0),edge_coords[1]))

            # path_final_ant = []
            # for i in range(coord_targets_ant.shape[0]):
            #     path_final_ant.append(np.where(np.all(np.equal(coord_ant[:, None], coord_targets_ant[i,:]), axis=2).any(axis=1))[0][0])

            # path_final_post =[]
            # for i in range(coord_targets_post.shape[0]):
            #     path_final_post.append(np.where(np.all(np.equal(coord_post[:, None], coord_targets_post[i,:]), axis=2).any(axis=1))[0][0])


            # coord_targets_ant_test = find_smoothed_path_geo(geoalg_ant,path_final_ant,1)
            # coord_targets_post_test = find_smoothed_path_geo(geoalg_post,path_final_post,1)


            # interpolation points
            n_boundary_nodes = round(max(len(path_ant), len(path_post)) * 8)*2

            orifice_triangulation = get_triangulated_area_between_two_contours(coord_targets_ant,coord_targets_post,n_boundary_nodes)
            orifice_area = orifice_triangulation.area

            if i == 0 and j == 0:
                orifice_triangulation_opt = orifice_triangulation
                orifice_area_opt = orifice_area
                combi_opt = [skip_n[i], skip_m[j]]
                coord_opt_ant = coord_targets_ant
                coord_opt_post = coord_targets_post
            else:
                if orifice_area < orifice_area_opt:
                    orifice_triangulation_opt = orifice_triangulation
                    orifice_area_opt = orifice_area
                    combi_opt = [skip_n[i], skip_m[j]]
                    coord_opt_ant = coord_targets_ant
                    coord_opt_post = coord_targets_post
                    # if good_solution:
                    #     good_solution_found = True

    # print(coord_opt_ant)
    # print(coord_opt_post)

    return orifice_triangulation_opt

def get_triangulated_area_between_two_contours(contour_1,contour_2,n_boundary_nodes):

    n = np.round(n_boundary_nodes/2).astype(int)
    contour_1_inter = interpolate_points(contour_1,np.linspace(0, 1, n).tolist())
    contour_2_inter = interpolate_points(contour_2,np.linspace(0, 1, n).tolist())
    vertices = np.concatenate((contour_1_inter, contour_2_inter), axis=0)

    connectivity_1 = np.vstack((np.arange(1, n), np.arange(2, n+1), np.arange(n+2, 2*n+1))).T-1
    connectivity_2 = np.vstack((np.arange(1, n), np.arange(n+2, 2*n+1), np.arange(n+1, 2*n))).T-1
    faces = np.concatenate((connectivity_1, connectivity_2), axis=0)

    orifice_triangulation = trimesh.Trimesh(vertices=vertices,
                           faces=faces)

    return orifice_triangulation

def get_boundary_faces_indices(mesh):
    mesh_new = trimesh.Trimesh(vertices=mesh.vertex_matrix(),
                            faces=mesh.face_matrix())
    index = trimesh.grouping.group_rows(mesh_new.edges_sorted, require_count=1)
    boundary_vertices = np.unique(mesh_new.edges[index].flatten())
    boundary_edges = mesh_new.edges[index]
    boundary_faces = mesh_new.faces[mesh_new.edges_face[index]]
    boundary_faces_indices = np.unique(mesh_new.edges_face[index])

    return boundary_faces_indices

def get_perimeter(mesh):
    boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    perimeter = 0.0
    for i in range(len(boundary_edges)):
        v0=boundary_edges[i][0]
        v1=boundary_edges[i][1]
        edge_length = np.linalg.norm(mesh.vertices[v0,:] - mesh.vertices[v1,:])
        perimeter = perimeter+edge_length
    return perimeter

def remesh_and_smooth_triangulation(file,iterations=3):

    meshset = ml.MeshSet()
    meshset.load_new_mesh(file)

    if iterations > 0:

        for i in range(iterations):

            meshset.set_selection_none()
            meshset.meshing_isotropic_explicit_remeshing(iterations = 30,adaptive = True,targetlen=ml.Percentage(1),featuredeg = 180)
            mesh_temp = meshset.current_mesh()

            bd_faces_indices = get_boundary_faces_indices(mesh_temp)
            for i in range(len(bd_faces_indices)):
                cond = '(fi == ' + str(bd_faces_indices[i]) + ') || (fsel == 1)'
                meshset.compute_selection_by_condition_per_face(condselect = cond)
            meshset.apply_selection_inverse(invfaces = True)


            meshset.apply_coord_laplacian_smoothing(stepsmoothnum = 50,boundary = True,selected = True)
            meshset.apply_coord_laplacian_smoothing(stepsmoothnum = 1,boundary = True,selected = False)

    mesh_remesh = meshset.current_mesh()
    mesh_new = trimesh.Trimesh(vertices=mesh_remesh.vertex_matrix(),
                            faces=mesh_remesh.face_matrix())

    return mesh_new


###############################################################################
# IMPORTANT NOTE
###############################################################################
'''
Code above can be changed or updated.

After changing code, comment all print commands, since these cause issues 
when executed in the shell. 

Yet, the print command in the last line "print(orifice_area[0])" enables the 
Python Interpreter to read the result of the computation.

'''
###############################################################################
# execution in shell
###############################################################################
# set input files (anterior and posterior leaflet meshes)
input_file_ant = "pymeshlab/shape_anterior.stl"
input_file_post = "pymeshlab/shape_posterior.stl"

# output folder and output orifice file name
output_folder = "/pymeshlab"
output_name = "mitral_valve_MVA_3D"

# compute and save orifice mesh
orifice_area = get_3D_orifice_area(input_file_ant,
                                   input_file_post,
                                   output_folder,
                                   output_name,
                                   save_mesh = False,
                                   accuracy_setting='closest_edge_pair',
                                   iter_smooth=3)

### save orifice area in csv file
# np.savetxt(os.path.join(output_folder,'mva_data.csv'), [p for p in zip(orifice_area)], delimiter=',', fmt='%s')

results = {
    "orifice_area" : orifice_area
}

###############################################################################
# print results as dictionary expressed as a string
print(orifice_area[0])

# EOF