import torch

from elasticsolid import *
from objectives import *
from harmonic_interpolator import *
from adjoint_sensitivity import *

from IPython import display
import time

from vis_utils import plot_torch_solid
from utils import *

torch.set_default_dtype(torch.float64)


class ShapeOptimizer():
    def __init__(self, solid, vt_surf, weight_reg=0.):
        
        
        # Elastic Solid with the initial rest vertices stored 
        self.solid = solid

        # Mesh info of solid
        bvNP, ivNP = get_boundary_and_interior(solid.v_rest.shape[0], to_numpy(solid.t))
        self.bvNP, self.ivNP = bvNP, ivNP
        self.beNP  = igl.edges(igl.boundary_facets(to_numpy(solid.t)))

        # Initialize Laplacian/Harmonic Interpolator
        v_init_rest = solid.v_rest.clone().detach()
        self.harm_int = HarmonicInterpolator(v_init_rest, solid.t, ivNP)

        # Initialize interior vertices with harmonic interpolation
        self.v_init_rest = self.harm_int.interpolate(v_init_rest[bvNP])
        solid.update_rest_shape(self.v_init_rest)

        # Define optimization params and their indices
        self.params_idx = torch.tensor(np.intersect1d(to_numpy(solid.free_idx), bvNP))
        params_init = v_init_rest[self.params_idx].reshape(-1,)
        self.params, self.params_prev  = params_init.clone(), params_init.clone() # At time step t, t-1

        # Target surface and Objectives
        self.vt_surf = vt_surf
        self.tgt_fit = ObjectiveBV(vt_surf, bvNP)
        self.neo_reg = ObjectiveReg(params_init.clone().detach(), self.params_idx, self.harm_int, weight_reg = weight_reg)
        
        # Compute equilibrium deformation
        self.solid.find_equilibrium()
        obj_init = self.tgt_fit.obj(solid.v_def.clone().detach())
        print("Initial objective: {:.4e}\n".format(obj_init))

        # Initialize grad
        self.grad = torch.zeros(size=(3 * self.params_idx.shape[0],))

        # BFGS book-keeping
        self.invB      = torch.eye(3 * self.params_idx.shape[0])
        self.grad_prev = torch.zeros(size=(3 * self.params_idx.shape[0],))
        
    def compute_gradient(self):
        '''
        Computes the full gradient including the forward simulation and regularization.

        Updated attributes:
        - grad : torch.tensor of shape (#params,)
        '''
        # dJ/dx from Target Fitting
        dJ_dx = self.tgt_fit.grad(self.solid.v_def.clone().detach())[self.solid.free_idx]

        self.grad = gradient_helper_autograd(self.solid, dJ_dx, self.params, self.params_idx, self.harm_int)
        
        # Add regularization gradient
        self.grad = self.grad + self.neo_reg.grad(self.solid, self.params)
        return self.grad
    
    def update_BFGS(self):
        '''
        Update BFGS hessian inverse approximation
        
        Updated attributes:
        - invB : torch.tensor of shape (#params, #params)
        '''
        sk = self.params - self.params_prev
        yk = self.grad - self.grad_prev
        self.invB = compute_inverse_approximate_hessian_matrix(sk.reshape(-1, 1), yk.reshape(-1, 1), self.invB)
    
    def reset_BFGS(self):
        '''
        Reset BFGS hessian inverse approximation to Identity
        
        Updated attributes:
        - invB : torch.tensor of shape (#params, #params)
        '''
        self.invB = torch.eye(3 * self.params_idx.shape[0])

    def set_params(self, params_in, verbose=False):
        '''
        Set optimization params to the input params_in
        
        Input:
        - params_in : Input params to set the solid to, torch.tensor of shape (#params,)
        - verbose : Boolean specifying the verbosity of the equilibrium solve
        
        Updated attributes:
        - solid : Elastic solid, specifically, the rest shape and consequently the equilibrium deformation
        '''
        # From the input params_in, interpolate all the rest vertices using harm_int
        v_search      = self.solid.v_rest.clone()
        v_search[self.params_idx, :] = params_in.reshape(-1, 3)
        v_search = self.harm_int.interpolate_fill(v_search)

        # Update solid using the new rest shape v_search - TODO
        
        # Update equilibrium deformation - TODO 
        
    def compute_obj(self):
        '''
        Compute Objective at the current params
        
        Output:
        - obj : Accumulated objective value
        '''

        obj = self.tgt_fit.obj(self.solid.v_def.clone().detach())
        obj += self.neo_reg.obj(self.solid, self.params)
        return obj

    def line_search_step(self, step_size_init, max_l_iter):
        '''
        Perform line search to take a step in the BFGS descent direction at the current optimization state
        
        Input:
        - step_size_init : Initial value of step size
        - max_l_iter : Maximum iterations of line search
        
        Updated attributes:
        - solid : Elastic solid, specifically, the rest shape and consequently the equilibrium deformation
        - params, params_prev : torch Tensor of shape (#params,)
        - grad_prev : torch Tensor of shape (#params,)

        Output:
        - obj_new : New objective value after taking the step
        - l_iter : Number of line search iterations taken
        '''

        step_size  = step_size_init
        # Compute previous objective for armijo rule
        obj_prev = self.compute_obj()
        success = False
        for l_iter in range(max_l_iter):
            step_size   *= 0.5
            
            # BFGS descent step - TODO
            descent_dir = ...
            params_search = self.params + step_size * descent_dir
            
            # Try taking a step
            try:
                # Save solid rest and def states in case steps fail
                rest_prev, def_prev = self.solid.v_rest.clone().detach(), self.solid.v_def.clone().detach()
                
                # Take a Step - TODO
                
                # Compute new objective - TODO
                obj_search = ...
                
                assert not torch.isnan(obj_search), "nan encountered"

                # Evaluate armijo condition - TODO
                armijo  = ... 
                if armijo or l_iter == max_l_iter-1:
                    self.params_prev = self.params.clone().detach() # BFGS bookkeeping
                    self.grad_prev = self.grad.clone() # BFGS bookkeeping
                    self.params, obj_new  = params_search, obj_search # params at t becomes params at t+1
                    success = True
                    break
                    
            # Fallback to previous in case we take a step that is too large
            except Exception as e:
                print("An exception occured: {}. \nHalving the step.".format(e))
                self.solid.update_rest_shape(rest_prev) # fall back to non NaN values
                self.solid.update_def_shape(def_prev)   # faster convergence fo the next equilibrium
                if l_iter == max_l_iter - 1:
                    return obj_prev, l_iter, success
        return obj_new, l_iter, success

    def optimize(self, step_size_init=1e-3, max_l_iter=10, n_optim_steps=10):
        '''
        Run BFGS to optimize over the objective J.
        
        Input:
        - step_size_init : Initial value of step size
        - max_l_iter : Maximum iterations of line search
        
        Updated attributes:
        - objectives : Tensor tracking objectives across optimization steps
        - grad_mags : Tensor tracking norms of the gradient across optimization steps
        '''

        self.objectives  = torch.zeros(size=(n_optim_steps+1,))
        self.grad_mags   = torch.zeros(size=(n_optim_steps,))

        self.objectives[0] = self.compute_obj()

        startTime = time.time()
        for i in range(n_optim_steps):
            # Update the gradients
            self.grad_mags[i] = torch.linalg.norm(self.compute_gradient())

            # Update quatities of BFGS
            if i >= 1:
                self.update_BFGS()
            
            # Line Search - TODO
            '''
            Updates self.objectives[i+1], l_iter, success on returning
            '''

            if not success:
                print("Line search can't find a step to take")
                return

            display.clear_output(wait=True)
            # Remaining time
            curr_time = (time.time() - startTime)
            rem_time  = (n_optim_steps - i - 1) / (i + 1) * curr_time
            print("Objective after {} optimization step(s): {:.4e}".format(i+1, self.objectives[i+1]))
            print("    Line search Iters: " + str(l_iter))
            print("Elapsed time: {:.1f}s. \nEstimated remaining time: {:.1f}s\n".format(curr_time, rem_time))
            
            # Plot the resulting mesh
            rot = np.array(
                [[1, 0, 0 ],
                [0, 0, 1],
                [0, -1, 0 ]]
            )
            aabb = np.max(to_numpy(self.solid.v_rest), axis=0) - np.min(to_numpy(self.solid.v_rest), axis=0)
            length_scale = np.mean(aabb)
            plot_torch_solid(self.solid, self.beNP, rot, length_scale, target_mesh=self.vt_surf)
            
            # # Early Termination
            # if (self.objectives[i] - self.objectives[i+1]) < 1e-3 * self.objectives[i]:
            #     print("Decrease regularization weight.")
            #     self.neo_reg.weight_reg *= 0.5
            #     invB = torch.eye(3 * self.params_idx.shape[0])
            
            # if (self.objectives[i] - self.objectives[i+1]) < 1e-5 * self.objectives[i]:
            #     print("Stop optimization due to non satisfactory relative progress on the objective.")
            #     break


def gradient_helper_autograd(solid, dJ_dx, params_tmp, params_idx, harm_int):
    '''
    Computes a pytorch computational flow to compute the gradient of the forward simulation through automatic differentiation
    '''
    # Adjoint state y
    adjoint   = compute_adjoint(solid, dJ_dx)    

    # Define the variable to collect the gradient of f_tot. y
    params_collect = params_tmp.clone()
    params_collect.requires_grad = True
    
    # Model f_tot.y as a differentiable pytorch function of params to collect gradient from auto_grad
    dot_prod = adjoint_dot_forces(params_collect, solid, adjoint, params_idx, harm_int)
    dot_prod.backward()
    dJ_dX = params_collect.grad.clone()
    params_collect.grad = None # Reset grad

    return dJ_dX
    
def adjoint_dot_forces(params, solid, adjoint, params_idx, harm_int):
    '''
    Input:
    - params    : array of shape (3*#params,)
    - solid     : an elastic solid to copy with the deformation at equilibrium
    - adjoint   : array of shape (3*nV,)
    - params_idx : parameters index in the vertex list. Has shape (#params,)
    - harm_int : harmonic interpolator
    
    Output:
    - dot_prod : dot product between forces at equilibrium and adjoint state vector
    '''
    
    # From params, compute the full rest state using the harmonic interpolation
    v_vol    = solid.v_rest.detach()
    v_vol[params_idx, :] = params.reshape(-1, 3)
    v_update = harm_int.interpolate_fill(v_vol)

    #Initialize a solid with this rest state and the same deformed state of the solid
    
    if "LinearElasticEnergy" in str(type(solid.ee)):
        ee_tmp   = LinearElasticEnergy(solid.ee.young, solid.ee.poisson)
    elif "NeoHookeanElasticEnergy" in str(type(solid.ee)):
        ee_tmp   = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
        
    solid_tmp = ElasticSolid(v_update, solid.t, ee_tmp, rho=solid.rho, 
                             pin_idx=solid.pin_idx, f_mass=solid.f_mass)
    solid_tmp.update_def_shape(solid.v_def.clone().detach())
    
    return adjoint.detach() @ (solid_tmp.f + solid_tmp.f_ext).reshape(-1,)

            
    
    

