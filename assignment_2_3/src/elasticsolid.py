import numpy as np
from numpy import linalg
from scipy import sparse
from Utils import *
import igl


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             ELASTIC SOLID CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ElasticSolid(object):

    def __init__(self, v_rest, t, ee, rho=1, pin_idx=[], f_mass=None):
        '''
        Input:
        - v_rest      : position of the vertices of the mesh (#v, 3)
        - t           : indices of the element's vertices (#t, 4)
        - ee          : elastic energy object that can be found in elasticenergy.py
        - rho         : mass per unit volume [kg.m-3]
        - pin_idx     : list or np array of vertex indices to pin
        - f_mass      : external force per unit mass (3,) [N.kg-1]
        '''

        self.v_rest   = v_rest.copy()
        self.v_def    = v_rest.copy()
        self.t        = t
        self.ee       = ee
        self.rho      = rho
        self.pin_idx  = np.array(pin_idx)
        self.f_mass   = f_mass.copy()
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

        self.energy_el  = None
        self.energy_ext = None

        self.f     = None
        self.f_vol = None
        self.f_ext = None

        self.make_free_indices_and_pin_mask()
        self.update_rest_shape(self.v_rest)
        self.update_def_shape(self.v_def)

    ## Utils ##

    def vertex_tet_sum(self, data):
        '''
        Distributes data specified at each tetrahedron to the neighboring vertices.
        All neighboring vertices will receive the value indicated at the corresponding tet position in data.

        Input:
        - data : np array of shape (#t,) or (4*#t,)

        Output:
        - data_sum : np array of shape (#v,), containing the summed data
        '''
        i = self.t.flatten('F')        # (4*#t,)
        j = np.arange(len(self.t))     # (#t,)
        j = np.tile(j, 4)              # (4*#t,)

        if len(data) == len(self.t):
            data = data[j]

        # Has shape (#v, #t)
        m = sparse.coo_matrix((data, (i, j)), (len(self.v_rest), len(self.t)))
        return np.array(m.sum(axis=1)).flatten()

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
        Construct Ds that has shape (#t, 3, 3), and its inverse Bm

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
        - W0     : np array of shape (#t,) containing the signed volume of each tet
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
        - W     : np array of shape (#t,) containing the signed volume of each tet
        '''
        pass
        
    def displace(self, v_disp):
        '''
        Displace the whole mesh so that v_def += v_disp

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)
        '''

        self.update_def_shape(self.v_def + v_disp)

    ## Energies ##

    def make_elastic_energy(self):
        '''
        This updates the elastic energy

        Updated attributes:
        - energy_el  : elastic energy of the system [J]
        '''
        pass

    def make_external_energy(self):
        '''
        This computes the external energy potential

        Updated attributes:
        - energy_ext : postential energy due to external forces [J]
        '''
        pass

    ## Forces ##

    def make_elastic_forces(self):
        '''
        This method updates the elastic forces stored in self.f (#v, 3)

        Updated attributes:
        - f  : elastic forces per vertex (#v, 3)
        - ee : elastic energy, some attributes should be updated
        '''
        pass

    def make_volumetric_and_external_forces(self):
        '''
        Convert force per unit mass to volumetric forces, then distribute
        the forces to the vertices of the mesh.

        Updated attributes:
        - f_vol : np array of shape (#t, 3) external force per unit volume acting on the tets
        - f_ext : np array of shape (#v, 3) external force acting on the vertices
        '''
        pass

    ## Force Differentials

    def compute_force_differentials(self, v_disp):
        '''
        This computes the differential of the force given a displacement dx,
        where df = df/dx|x . dx = - K(x).dx. Where K(x) is the stiffness matrix (or Hessian)
        of the solid. Note that the implementation doesn't need to construct the stiffness matrix explicitly.

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df : force differentials at the vertices of the mesh (#v, 3)

        Updated attributes:
        - ee : elastic energy, some attributes should be updated
        '''

        # First compute the differential of the Jacobian
        
        
        # Then update differential quantities in self.ee


        # Compute the differential of the forces


        return np.zeros_like(self.v_rest)
    
    def equilibrium_step(self, step_size_init = 2, max_l_iter = 20, c1 = 1e-4):
        '''
        This function displaces the whole solid to the next deformed configuration
        using a Newton-CG step.

        Updated attributes:
        - LHS : The hessian vector product
        - RHS : Right hand side for the conjugate gradient linear solve
        Other than them, only attributes updated by displace(self, v_disp) should be changed
        '''

        # Define LHS
        def LHS(dx):
            '''
            Should implement the Hessian-Vector Product L(dx), and take care of pinning constraints
            as described in the handout.
            '''
            return None
        
        self.LHS = LHS # Save to class for testing
        
        # Define RHS
        RHS = ...
        
        self.RHS = RHS # Save to class for testing
        
        # Use conjugate gradient to find a descent direction
        # (see conjugate_gradient in Utils.py)
        dx_CG = ... # shape (#v, 3)

        
        # Run line search on the direction
        step_size = step_size_init
        ft = self.f + self.f_ext
        energy_tot_prev = self.energy_el + self.energy_ext
        for l_iter in range(max_l_iter):
            step_size *= 0.5 # DO NOT CHANGE
            dx_search = dx_CG * step_size
            self.displace(dx_search)
            
            # Current force norm for unpinned vertices
            g = np.linalg.norm(...)

            energy_tot_tmp = self.energy_el + self.energy_ext
            
            #Make sure you only use the gradient of the unpinned vertices
            # USE parameter c1 = 1e-4 (from the function arguments) when you submit
            armijo = ...
            
            if armijo or l_iter == max_l_iter-1:
                print("Energy: " + str(energy_tot_tmp) + " Force norm: " + str(g) + " Line search Iters: " + str(l_iter))
                break
            else:
                self.displace(-dx_search)
