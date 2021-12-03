import igl
import numpy as np
import math
import numpy.linalg as la
from tracer_helper import rotate_vector

def asymptotic_path(idx, mesh, num_steps, step_size, first_principal_direction, num_neighbors, sampling_dist=0):
    '''Computes both tracing direction (backward and forward) following an asymptotic path.
    Try to compute an asymptotic path with both tracing directions 

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - num_neighbors : int
        Number of closest vertices to consider for avering principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of unique points representing a full asymptotic path.
    - A : np.array (n,)
        The ordered set of deviated angles in degrees calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    '''
        
    P = None
    A = None
    PP = None

    return P, A, PP

def trace(idx, mesh, num_steps, step_size, first_principal_direction, trace_backwards, num_neighbors, sampling_dist):
    '''Computes one tracing direction following an asymptotic path.
    Try to compute the points on the asymptotic path.

    Inputs:
    - idx : int
        The index of the vertex on the mesh to start tracing.
    - mesh : Mesh
        The mesh for tracing.
    - num_steps : int
        The number of tracing steps.
    - step_size : int
        The size of the projection at each tracing step.
    - first_principal_direction : bool
        Indicator for using the first principal curvature to do the tracing.
    - trace_backwards : bool
        Indicator for mirroring the deviated angle
    - num_neighbors : int
        Number of closest vertices to consider for avering principal curvatures.
    - sampling_distance : float
        The distance to sample points on the path (For the design interface - Don't need to be define).
            
    Outputs:
    - P : np.array (n, 3)
        The ordered set of points representing one tracing direction.
    - A : np.array (n,)
        The ordered set of deviated angles calculated for the asymptotic directions.
    - PP : np.array (m,3)
        The ordered set of points laying on the path spaced with a given distance (For the design interface).
    '''

    P = np.empty((0, 3), float)
    PP = np.empty((0,3), float)
    A = np.array([],float)

    #Get the data of the first vertex in the path
    pt = mesh.V[idx]

    # Store partial distance (For the design interface)
    partial_dist = 0

    while len(P) < num_steps:

        # Add the current point to the path
        P = np.append(P, np.array([pt]), axis=0)

        # TODO: Get the averaged principal curvature directions & values
        k1_aver, k2_aver, v1_aver, v2_aver, n_aver = averaged_principal_curvatures(pt, mesh, num_neighbors)

        # TODO: Calculate deviation angle (theta) based on principal curvature values
        theta = None
            
        # Store theta
        A = np.append(A, np.array([theta]), axis=0)

        # TODO: Mirror the angle for tracing backwards. Use trace_backwards indicator

        # TODO: Rotate principal curvature direction to get asymptotic direction. Use first_principal_direction indicator
        a_dir = None

        # TODO: Check for anticlastic surface-regions

        # TODO: Check for valid asymptotic direction and unitize

        # TODO: Prevent the tracer to go in the opposite direction

        # TODO: Scale the asymptotic direction to the given step-size

        # TODO: Compute edge-point
        edge_point, is_boundary_edge = find_edge_point(mesh, pt, a_dir)

        # TODO: Check for boundaries

        # TODO: Check for duplicated points

        # Store sampling points (For the design interface)
        if sampling_dist>0:
            partial_dist += la.norm(edge_point-pt)
            if partial_dist >= sampling_dist :
                partial_dist = 0
                PP = np.append(PP, np.array([edge_point]), axis=0)

        pt = edge_point

    return P, A, PP
    
def averaged_principal_curvatures(pt, mesh, num_neighbors=2, eps=1e-6):
    '''Computes inverse weighted distance average of principal curvatures of a given mesh-point
       on the basis of the two closest vertices.
    Try to compute values, directions and normal at the given query point.

    Inputs:
    - pt : np.array (3,)
        The query point position.
    - mesh : Mesh
        The mesh for searching nearest vertices.
    - num_neighbors : int
        Number of closest vertices to consider for avering.
    - eps : float
        The distance tolerance to consider whether the given point and a mesh-vertex are coincident.
            
    Outputs:
    - k_1 : np.array (n)
        The min principal curvature average at the given query point.
    - k_2 : np.array (n)
        The max principal curvature average at the given query point.
    - v1_aver : np.array (3,)
        The unitized min principal curvature direction average at the given query point.
    - v2_aver : np.array (3,)
        The unitized max principal curvature direction average at the given query point.
    - n_aver : np.array (3,)
        The unitized normal average at the given query point.
    '''

    # Get the closest vertices and distances to the query point
    # Use these data to compute principal curvature weighted averages.
    dist, neighbors = mesh.get_closest_vertices(pt, num_neighbors)

    v1_aver = None
    v2_aver = None
    k1_aver = None
    k2_aver = None
    n_aver = None

    return k1_aver, k2_aver, v1_aver, v2_aver, n_aver

def find_edge_point(mesh, a_orig, a_dir):
    '''Computes the point where a mesh-edge intersects with the asymptotic projection.
    Try to compute the edge-point resulting from this intersection.

    Inputs:
    - mesh : Mesh
        The mesh for searching edge intersections.
    - a_orig : np.array (3,)
        The start position of the asymptotic projection.
    - a_dic : np.array (3,)
        The direction of the asymptotic projection.
            
    Outputs:
    - edge_point : np.array (3,)
        The position of the edge-point.
    - is_boundary_point : bool
        Indicator for whether the edge-point is at the boundary of the mesh.
    '''

    # Get the closest face-index and mesh-point (point laying on the mesh)
    proj_pt = a_orig+a_dir
    face_index, mesh_point = mesh.get_closest_mesh_point(proj_pt)

    # Update the projection vector with the position of the mesh-point
    a_dir = mesh_point - a_orig

    # If the mesh-point is equal to the starting point, return flag for boundary vertex.
    if la.norm(a_dir)==0:
        return mesh_point, True

    # Unitize projection vector
    a_dir /= la.norm(a_dir)

    # Initialize variables
    edge_point = mesh_point
    is_boundary_point = False
    prev_projection_param = 0
 
    # Find the required edge-point by computing intersections between the edge-segments of the face and the asymptotic-segment. 
    # Different intersection events need to be considered. 
    edges = mesh.face_edges[face_index]
    for e_idx in edges:

        e = mesh.edge_vertices[e_idx]
        e_orig = mesh.V[e[0]]
        e_dir = mesh.V[e[1]]-e_orig
        is_boundary_edge = np.any(mesh.edge_faces[e_idx]== -1)

        edge_param, projection_param, intersection = intersection_event(e_orig, e_dir, a_orig, a_dir)

        # TODO: Find the edge-point

    return edge_point, is_boundary_point

def intersection_event(a_orig, a_dir, b_orig, b_dir, eps=1e-6):
    '''Computes the intersection event between segments A and B.
    Try to compute the intersection event.

    Inputs:
    - a_orig : np.array (3,)
        The start position of segment A.
    - a_dic : np.array (3,)
        The direction of the segment A.
    - b_orig : np.array (3,)
        The start position of segment B.
    - b_dic : np.array (3,)
        The direction of the segment B.
    - eps : float
        The tolerance for determining intersections.
            
    Outputs:
    - t : float
        The parameter on segment A where the intersection occurred.
    - u : float
        The parameter on segment B where the intersection occurred.
    - E  : int
        Indicator for the type of intersection event. 
        Returns 0 if the intersection occurred within the domain [0,1] of both segments. 
        Returns 1 if the intersection occured outside the domain of at least one the segments.
        Returns 2 for collinearity.
    '''
    u = None
    t = None
    E = None

    return t, u, E
