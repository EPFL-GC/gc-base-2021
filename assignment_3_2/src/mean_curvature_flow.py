import laplacian_utils
import scipy as sp
from scipy.sparse.linalg import spsolve
import igl

from utils import normalize_area, has_zero_area_triangle
import copy 
import numpy as np
import numpy.linalg as la

class MCF():
    def __init__(self, num_bdry_vx, num_intr_vx):
        '''
        Inputs:
        - num_bdry_vx : int
            The first num_bdry_vx vertices in v are boundary vertices in the mesh.
        - num_intr_vx : int
            The number of interior vertices in the mesh.
        '''
        self.num_bdry_vx = num_bdry_vx
        self.num_intr_vx = num_intr_vx

        self.L = None # Laplacian matrix.
        self.M = None # Mass matrix.
        self.average_mean_curvature = 0 # The average mean curvature value of the mesh.

    def update_system(self, v, f):
        '''
        Update the member variables in the class, including the mass matrix,  the Laplacian matrix, and the average mean curvature value of the mesh.
        '''

        self.M = ...
        self.L = ...

        # Update current average mean curvature.
        self.average_mean_curvature = ... 
        # Normalize the average mean curvature using the length of the bounding box diagonal of the mesh.
        self.average_mean_curvature /= igl.bounding_box_diagonal(v)

    def solve_laplace_equation(self, v, f):
        '''
        Solve the Laplace equation for the current mesh. Update the vertex positions with the solution.
        '''
        # Construct the LHS and RHS of the linear solve. 
        lhs = ...
        rhs = ...

        # Solve linear system.
        # ... = spsolve(lhs, rhs)

        # Update the vertex positions using the solution of the linear solve. 
        # v ...

    def meet_stopping_criteria(self, mean_curvature_list, epsilon1, epsilon2):
        '''
        Stopping criteria for mean curvature flow.
        '''
        if (len(mean_curvature_list) < 2):
            return False
        # If the changes in the iteration is smaller than epsilon1, terminate the flow. 
        if (... < epsilon1):
            print("Insufficient improvement from the previous iteration!")
            return True
        # If the average mean curvature value of the mesh is sufficiently small, terminate the flow.
        if (... < epsilon2):
            print("Sufficiently small average mean curvature value!")
            return True
        return False

    def run_mean_curvature_flow(self, v, f, max_iter, epsilon1, epsilon2):
        '''
        Running mean curvature flow by iteratively solving the Laplace equation.
        '''
        vs = [copy.deepcopy(v)]
        average_mean_curvature_list = []
        i = 0

        while i < max_iter and not has_zero_area_triangle(v, f) and not self.meet_stopping_criteria(average_mean_curvature_list, epsilon1, epsilon2):
            
            '''
            Your code here.
            '''
            if self.num_bdry_vx == 0:
                normalize_area(v, f)
            
            vs.append(copy.deepcopy(v))
            average_mean_curvature_list.append(self.average_mean_curvature)
            i += 1

        return vs, average_mean_curvature_list
