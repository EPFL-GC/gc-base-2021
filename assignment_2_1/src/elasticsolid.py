import numpy as np
from numpy import linalg
from scipy import sparse
import igl


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v_rest, t, rho=1, pin_idx=[]):
        '''
        Input:
        - v_rest      : position of the vertices of the mesh (#v, 3)
        - t           : indices of the element's vertices (#t, 4)
        - rho         : mass per unit volume [kg.m-3]
        - pin_idx     : list of vertex indices to pin
        '''

        self.v_rest   = v_rest.copy()
        self.v_def    = v_rest.copy()
        self.t        = t
        self.rho      = rho
        self.pin_idx  = pin_idx
        self.free_idx = None
        self.pin_mask = None
        
        self.W0 = None
        self.Dm = None
        self.Bm = None
        self.rest_barycenters = None

        self.W  = None
        self.Ds = None
        self.F  = None
        self.def_barycenters = None

        self.make_free_indices_and_pin_mask()
        self.update_rest_shape(self.v_rest)
        self.update_def_shape(self.v_def)

    ## Precomputation ##

    def make_free_indices_and_pin_mask(self):
        '''
        Should list all the free indices and the pin mask.

        Updated attributes:
        - free_index : np array of shape (#free_vertices,) containing the list of unpinned vertices
        - pin_mask   : np array of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''
        pass

    ## Methods related to rest quantities ##

    def make_rest_barycenters(self):
        '''
        Construct the barycenters of the undeformed configuration

        Updated attributes:
        - rest_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''
        pass

    def make_rest_shape_matrices(self):
        '''
        Construct Dm that has shape (#t, 3, 3), and its inverse Bm

        Updated attributes:
        - Dm : np array of shape (#t, 3, 3) containing the shape matrix of each tet
        - Bm : np array of shape (#t, 3, 3) containing the inverse shape matrix of each tet
        '''
        pass

    def update_rest_shape(self, v_rest):
        '''
        Updates the vertex position, the shape matrices Dm and Bm, the volumes W0,
        and the mass matrix at rest

        Input:
        - v_rest : position of the vertices of the mesh at rest state (#v, 3)

        Updated attributes:
        - v_rest : np array of shape (#v, 3) containing the position of each vertex at rest
        - W0     : np array of shape (#t, 3) containing the volume of each tet
        '''
        pass

    ## Methods related to deformed quantities ##

    def make_def_barycenters(self):
        '''
        Construct the barycenters of the deformed configuration

        Updated attributes:
        - def_barycenters : np array of shape (#t, 3) containing the position of each tet's barycenter
        '''
        pass

    def make_def_shape_matrices(self):
        '''
        Construct Ds that has shape (#t, 3, 3)

        Updated attributes:
        - Ds : np array of shape (#t, 3, 3) containing the shape matrix of each tet
        '''
        pass

    def make_jacobians(self):
        '''
        Compute the current Jacobian of the deformation

        Updated attributes:
        - F : np array of shape (#t, 3, 3) containing Jacobian of the deformation in each tet
        '''
        pass

    def update_def_shape(self, v_def):
        '''
        Updates the vertex position, the Jacobian of the deformation, and the 
        resulting elastic forces.

        Input:
        - v_def : position of the vertices of the mesh (#v, 3)

        Updated attributes:
        - v_def : np array of shape (#v, 3) containing the position of each vertex after deforming the solid
        - W     : np array of shape (#t, 3) containing the volume of each tet
        '''
        pass
        
    def displace(self, v_disp):
        '''
        Displace the whole mesh so that v_def += v_disp

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)
        '''
        self.update_def_shape(self.v_def + v_disp)
