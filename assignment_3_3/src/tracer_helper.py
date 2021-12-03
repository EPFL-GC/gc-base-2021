import igl
import numpy as np
import numpy.linalg as la
import scipy.spatial as ds
from meshplot import plot

from fitting import compute_mesh_principal_curvatures

class Mesh():
    def __init__(self, *args):
        '''
        Inputs:
        - filename : str
            Path to mesh file.
        '''
        if isinstance(args[0], str):
            # Build vertices and faces from file
            self.V, self.F = igl.read_triangle_mesh(args[0])
            self.build()
        elif len(args)==2:
            self.V = args[0]
            self.F = args[1]
            self.build()

    def build(self):
        # Calculate principal curvatures
        self.K1, self.K2, self.V1, self.V2  = compute_mesh_principal_curvatures(self.V, self.F)

        # Calculate normals
        self.N = igl.per_vertex_normals(self.V, self.F)

        # Build topology
        self.edge_vertices, self.face_edges, self.edge_faces = igl.edge_topology(self.V,self.F)

        # Build KDTree for topologic queries
        self.kd_tree = ds.KDTree(self.V)
    
    def get_closest_vertices(self, point, numNeighbors):
        return self.kd_tree.query(point,numNeighbors) 

    def get_closest_mesh_point(self, point):
        D,F,P = igl.point_mesh_squared_distance(point, self.V, self.F)
        return F, P

def rotate_vector(v, ang, v1, v2, n):
    v1 /= la.norm(v1, axis = -1)
    v2 /= la.norm(v2, axis = -1)
    sign = np.sign(np.dot(np.cross(v1, v2), n.transpose()))
    v2 *= sign * -1

    v_arr = igl.rotate_vectors(np.array(v, ndmin=2), np.array([ang], ndmin=2), np.array(v1, ndmin=2), np.array(v2, ndmin=2))
    return v_arr  
