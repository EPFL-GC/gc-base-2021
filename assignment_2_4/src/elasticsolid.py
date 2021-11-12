import numpy as np
import torch
import igl
from utils import *

torch.set_default_dtype(torch.float64)

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                        ELASTIC SOLID CLASS (using PyTorch)
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
        - pin_idx     : list or torch tensor of vertex indices to pin
        - f_mass      : external force per unit mass (3,) [N.kg-1]
        '''

        self.v_rest   = v_rest
        self.v_def    = v_rest
        self.t        = t
        self.ee       = ee
        self.rho      = rho
        if not isinstance(pin_idx, torch.Tensor):
            pin_idx  = torch.tensor(pin_idx)
        self.pin_idx = pin_idx
        self.f_mass   = f_mass
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

        self.f       = None
        self.f_vol   = None
        self.f_ext   = None
        self.f_point = torch.zeros_like(v_rest)

        self.dF = None

        self.make_free_indices_and_pin_mask()
        self.update_rest_shape(self.v_rest)
        self.update_def_shape(self.v_def)

    ## Utils ##

    def vertex_tet_sum(self, data):
        '''
        Distributes data specified at each tetrahedron to the neighboring vertices.
        All neighboring vertices will receive the value indicated at the corresponding tet position in data.

        Input:
        - data     : torch array of shape (#t,)

        Output:
        - data_sum : torch array of shape (#v,), containing the summed data
        '''
        i = self.t.T.flatten()         # (4*#t,)
        j = torch.arange(self.t.shape[0]) # (#t,)
        j = torch.tile(j, (4,))         # (4*#t,)

        # Has shape (#v, #t), a bit less efficient than using sparse matrices
        m = torch.zeros(size=(self.v_rest.shape[0], self.t.shape[0]), dtype=torch.float64)
        m[i, j] = data

        return torch.sum(m, dim=1)

    ## Precomputation ##

    def make_free_indices_and_pin_mask(self):
        '''
        Should list all the free indices and the pin mask.

        Updated attributes:
        - free_index : torch tensor of shape (#free_vertices,) containing the list of unpinned vertices
        - pin_mask   : torch tensor of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''

        vi            = torch.arange(self.v_rest.shape[0])
        pin_filter    = ~torch.isin(vi, self.pin_idx)
        self.free_idx = vi[pin_filter]

        self.pin_mask = torch.tensor([idx not in self.pin_idx 
                                   for idx in range(self.v_rest.shape[0])]).reshape(-1, 1)

    ## Methods related to rest quantities ##

    def make_rest_barycenters(self):
        '''
        Construct the barycenters of the undeformed configuration

        Updated attributes:
        - rest_barycenters : torch tensor of shape (#t, 3) containing the position of each tet's barycenter
        '''

        self.rest_barycenters = torch.einsum('ijk -> ik', self.v_rest[self.t]) / 4

    def make_rest_shape_matrices(self):
        '''
        Construct Ds that has shape (#t, 3, 3), and its inverse Bm

        Updated attributes:
        - Dm : torch tensor of shape (#t, 3, 3) containing the shape matrix of each tet
        - Bm : torch tensor of shape (#t, 3, 3) containing the inverse shape matrix of each tet
        '''

        e1 = (self.v_rest[self.t[:, 0]] - self.v_rest[self.t[:, 3]])
        e2 = (self.v_rest[self.t[:, 1]] - self.v_rest[self.t[:, 3]])
        e3 = (self.v_rest[self.t[:, 2]] - self.v_rest[self.t[:, 3]])
        Ds = torch.stack((e1, e2, e3), dim=2)
        self.Dm = Ds
        self.Bm = torch.linalg.inv(self.Dm)

    def update_rest_shape(self, v_rest):
        '''
        Updates the vertex position, the shape matrices Dm and Bm, the volumes W0,
        and the mass matrix at rest

        Input:
        - v_rest : position of the vertices of the mesh at rest state (#v, 3)

        Updated attributes:
        - v_rest : torch tensor of shape (#v, 3) containing the position of each vertex at rest
        - W0     : torch tensor of shape (#t,) containing the signed volume of each tet
        '''

        self.v_rest = v_rest
        self.make_rest_barycenters()
        self.make_rest_shape_matrices()
        self.W0 = - torch.linalg.det(self.Dm) / 6
        self.make_volumetric_and_external_forces()
        self.update_def_shape(self.v_def)

    ## Methods related to deformed quantities ##

    def make_def_barycenters(self):
        '''
        Construct the barycenters of the deformed configuration

        Updated attributes:
        - def_barycenters : torch tensor of shape (#t, 3) containing the position of each tet's barycenter
        '''

        self.def_barycenters = torch.einsum('ijk -> ik', self.v_def[self.t]) / 4

    def make_def_shape_matrices(self):
        '''
        Construct Ds that has shape (#t, 3, 3)

        Updated attributes:
        - Ds : torch tensor of shape (#t, 3, 3) containing the shape matrix of each tet
        '''

        e1 = (self.v_def[self.t[:, 0]] - self.v_def[self.t[:, 3]])
        e2 = (self.v_def[self.t[:, 1]] - self.v_def[self.t[:, 3]])
        e3 = (self.v_def[self.t[:, 2]] - self.v_def[self.t[:, 3]])
        Ds = torch.stack((e1, e2, e3), dim=2)
        self.Ds = Ds

    def make_jacobians(self):
        '''
        Compute the current Jacobian of the deformation

        Updated attributes:
        - F : torch tensor of shape (#t, 3, 3) containing Jacobian of the deformation in each tet
        '''

        self.F = torch.einsum('lij,ljk->lik', self.Ds, self.Bm)

    def update_def_shape(self, v_def):
        '''
        Updates the vertex position, the Jacobian of the deformation, and the 
        resulting elastic forces.

        Input:
        - v_def : position of the vertices of the mesh (#v, 3)

        Updated attributes:
        - v_def : torch tensor of shape (#v, 3) containing the position of each vertex after deforming the solid
        - W     : torch tensor of shape (#t,) containing the signed volume of each tet
        '''

        # Can only change the unpinned ones
        self.v_def = (~self.pin_mask) * self.v_rest + self.pin_mask * v_def
        self.make_def_barycenters()
        self.make_def_shape_matrices()
        self.make_jacobians()
        self.W = - torch.linalg.det(self.Ds) / 6
        self.make_elastic_forces()
        self.make_elastic_energy()
        self.make_external_energy()
        
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
        self.ee.make_strain_tensor(self.F)
        self.ee.make_energy_density(self.F)
        self.energy_el = torch.sum(self.W0 * self.ee.psi)

    def make_external_energy(self):
        '''
        This computes the external energy potential

        Updated attributes:
        - energy_ext : postential energy due to external forces [J]
        '''

        self.energy_ext  = - torch.einsum('i, ij, ij->', self.W0, self.f_vol, self.def_barycenters - self.rest_barycenters)
        
    ## Forces ##

    def make_elastic_forces(self):
        '''
        This method updates the elastic forces stored in self.f (#v, 3)

        Updated attributes:
        - f  : elastic forces per vertex (#v, 3)
        - ee : elastic energy, some attributes should be updated
        '''

        # First update strain/stress tensor, stored in self.ee
        self.ee.make_strain_tensor(self.F)
        self.ee.make_piola_kirchhoff_stress_tensor(self.F)

        # H[el] = - W0[el]*P.Bm[el]^T
        H = torch.einsum('lij,ljk->lik', self.ee.P, torch.swapaxes(self.Bm, 1, 2))
        H = - torch.einsum('i,ijk->ijk', self.W0, H)

        # Extract forces from H of shape (#t, 3, 3)
        # We look at each component separately, then stack them in a vector of shape (4*#t,)
        # Then we distribute the contributions to each vertex
        fx = self.vertex_tet_sum(torch.hstack((H[:, 0, 0], H[:, 0, 1], H[:, 0, 2],
                                            -H[:, 0, 0] - H[:, 0, 1] - H[:, 0, 2])))
        fy = self.vertex_tet_sum(torch.hstack((H[:, 1, 0], H[:, 1, 1], H[:, 1, 2],
                                            -H[:, 1, 0] - H[:, 1, 1] - H[:, 1, 2])))
        fz = self.vertex_tet_sum(torch.hstack((H[:, 2, 0], H[:, 2, 1], H[:, 2, 2],
                                            -H[:, 2, 0] - H[:, 2, 1] - H[:, 2, 2])))
        
        # We stack them in a tensor of shape (#v, 3)
        self.f = torch.column_stack((fx, fy, fz))

    def add_point_load(self, v_idx, f_point):
        '''
        Input:
        - v_idx   : tensor of size (n_idx,) containing the list of vertex ids
        - f_point : tensor of size (n_idx, 3) containing force to add at each vertex
        
        Updated attributes:
        - f_point : tensor of size (#v, 3)
        '''
        pass
        
    def make_volumetric_and_external_forces(self):
        '''
        Convert force per unit mass to volumetric forces, then distribute
        the forces to the vertices of the mesh.

        Updated attributes:
        - f_vol : torch tensor of shape (#t, 3) external force per unit volume acting on the tets
        - f_ext : torch tensor of shape (#v, 3) external force acting on the vertices
        '''
        self.f_vol = torch.tile((self.rho * self.f_mass), (self.t.shape[0], 1)) # (#t, 3)
        int_f_vol  = torch.einsum('i, ij -> ij', self.W0, self.f_vol)

        # from (#t,) to (4*#t,)
        j = torch.arange(len(self.t))
        j = torch.tile(j, (4,))
        int_f_vol_tiled = int_f_vol[j]
        
        f_ext_x = self.vertex_tet_sum(int_f_vol_tiled[:, 0])
        f_ext_y = self.vertex_tet_sum(int_f_vol_tiled[:, 1])
        f_ext_z = self.vertex_tet_sum(int_f_vol_tiled[:, 2])
        self.f_ext  = torch.stack((f_ext_x, f_ext_y, f_ext_z), dim=1) / 4
        

    ## Force Differentials

    def compute_force_differentials(self, v_disp):
        '''
        This computes the differential of the force given a displacement dx,
        where df = df/dx|x . dx = - K(x).dx. The matrix vector product K(x)w
        is then given by the call self.compute_force_differentials(-w).

        Input:
        - v_disp : displacement of the vertices of the mesh (#v, 3)

        Output:
        - df : force differentials at the vertices of the mesh (#v, 3)

        Updated attributes:
        - ee : elastic energy, some attributes should be updated
        '''

        # Compute the displacement differentials
        d1 = (v_disp[self.t[:, 0]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        d2 = (v_disp[self.t[:, 1]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        d3 = (v_disp[self.t[:, 2]] - v_disp[self.t[:, 3]]).reshape(-1, 3, 1)
        dDs = torch.cat((d1, d2, d3), dim=2)
        
        # Differential of the Jacobian
        dF = torch.einsum('lij,ljk->lik', dDs, self.Bm)

        # Differential of the stress tensor (uses the current stress tensor)
        self.ee.make_differential_strain_tensor(self.F, dF)
        self.ee.make_differential_piola_kirchhoff_stress_tensor(self.F, dF)
        
        # Differential of the forces
        dH = torch.einsum('lij,ljk->lik', self.ee.dP, torch.swapaxes(self.Bm, 1, 2))
        dH = - torch.einsum('i,ijk->ijk', self.W0, dH)

        # Same as for the elastic forces
        dfx = self.vertex_tet_sum(torch.hstack((dH[:, 0, 0], dH[:, 0, 1], dH[:, 0, 2],
                                             -dH[:, 0, 0] - dH[:, 0, 1] - dH[:, 0, 2])))
        dfy = self.vertex_tet_sum(torch.hstack((dH[:, 1, 0], dH[:, 1, 1], dH[:, 1, 2],
                                             -dH[:, 1, 0] - dH[:, 1, 1] - dH[:, 1, 2])))
        dfz = self.vertex_tet_sum(torch.hstack((dH[:, 2, 0], dH[:, 2, 1], dH[:, 2, 2],
                                             -dH[:, 2, 0] - dH[:, 2, 1] - dH[:, 2, 2])))
        
        # We stack them in a tensor of shape (#v, 3)
        return torch.column_stack((dfx, dfy, dfz))

    def equilibrium_step(self, verbose=False):
        '''
        This function displaces the whole solid to the next deformed configuration
        using a Newton-CG step.

        Input:
        - verbose : whether or not to display quantities

        Updated attributes:
        - LHS : The hessian vector product
        - RHS : Right hand side for the conjugate gradient linear solve
        Other than them, only attributes updated by displace(self, v_disp) should be changed
        '''

        dx0s = torch.zeros_like(self.v_rest)
        
        # Define LHS
        def LHS(dx):
            '''
            Should implement the Hessian-Vector Product L(dx), and take care of pinning constraints
            as described in the handout.
            '''
            dx0s[self.free_idx] = dx.reshape(-1,3)
            df0s = - self.compute_force_differentials(dx0s)
            return df0s[self.free_idx, :].reshape(-1,)
        self.LHS = LHS # Save to class for testing
        
        # Define RHS
        ft  = self.f + self.f_ext
        RHS = ft[self.free_idx, :].reshape(-1,)
        self.RHS = RHS # Save to class for testing

        dx   = conjugate_gradient(LHS, RHS)
        dx0s[self.free_idx] = dx.reshape(-1, 3)
        
        # Run line search on the direction
        step_size = 2
        ft_free = RHS
        g_old   = torch.linalg.norm(ft_free)
        max_l_iter = 20
        for l_iter in range(max_l_iter):
            step_size *= 0.5
            dx_search = dx0s * step_size
            energy_tot_prev = self.energy_el + self.energy_ext
            self.displace(dx_search)
            ft_new = (self.f_ext + self.f)[self.free_idx].reshape(-1,)
            g      = torch.linalg.norm(ft_new)

            energy_tot_tmp = self.energy_el + self.energy_ext
            armijo         = energy_tot_tmp < energy_tot_prev - 1e-4*step_size*torch.sum(dx.reshape(-1,)*ft_free)
            
            if armijo or l_iter == max_l_iter-1:
                if verbose:
                    print("Energy: " + str(energy_tot_tmp) + " Force residual norm: " + str(g) + " Line search Iters: " + str(l_iter))
                break
            else:
                self.displace(-dx_search)

    def find_equilibrium(self, n_steps=100, thresh=1., verbose=False):
        '''
        Input:
        - n_steps : maximum number of optimization steps
        - thresh  : threshold on the force value [N]
        '''
        for i in range(n_steps):
            # Take a Newton-CG step
            self.equilibrium_step(verbose=verbose)
            assert not torch.isnan(self.energy_el)

            # Measure the force residuals
            residuals_tmp = torch.linalg.norm((self.f + self.f_ext)[self.free_idx, :])
        
            if residuals_tmp < thresh:
                break
        if verbose: print("Final residuals (equilibrium): {:.2e}".format(to_numpy(residuals_tmp)))

def vertex_tet_sum(v, t, data):
    '''
    Distributes data specified at each tetrahedron to the neighboring vertices.
    All neighboring vertices will receive the value indicated at the corresponding tet position in data.

    Input:
    - num_vert : number of vertices
    - t        : connectivity of the volumetric mesh (#t, 4)
    - data     : torch array of shape (#t,)

    Output:
    - data_sum : torch array of shape (#v,), containing the summed data
    '''
    i = t.T.flatten()         # (4*#t,)
    j = torch.arange(t.shape[0]) # (#t,)
    j = torch.tile(j, (4,))         # (4*#t,)

    # Has shape (#v, #t), a bit less efficient than using sparse matrices
    m = torch.zeros(size=(v.shape[0], t.shape[0]), dtype=torch.float64)
    m[i, j] = data

    return torch.sum(m, dim=1)