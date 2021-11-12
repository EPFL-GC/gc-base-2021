import torch
import igl
import numpy as np
import matplotlib.pyplot as plt
import meshplot as mp
import time
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

torch.set_default_dtype(torch.float64)

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

def get_boundary_and_interior(vlen, t):
    bv = np.unique(igl.boundary_facets(t))
    vIdx  = np.arange(vlen)
    iv  = vIdx[np.invert(np.in1d(vIdx, bv))]
    return bv, iv


def conjugate_gradient(L_method, b):
        '''
        Finds an inexact Newton descent direction using Conjugate Gradient (CG)
        Solves partially L(x) = b where A is positive definite, using CG.
        The method should be implemented to check whether the added direction is
        an ascent direction or not, and whether the residuals are small enough.
        Details can be found in the handout.

        Input:
        - L_method : a method that computes the Hessian vector product. It should
                     take an array of shape (n,) and return an array of shape (n,)
        - b        : right hand side of the linear system (n,)

        Output:
        - p_star : torch array of shape (n,) solving the linear system approximately
        '''

        n = b.shape[0]
        p_star    = torch.zeros_like(b)
        residual  = - b
        direction = - residual.clone()
        grad_norm = torch.linalg.norm(b)

        # Reusable quantities
        L_direction    = L_method(direction)  # L d
        res_norm_sq    = torch.sum(residual ** 2, dim=0)  # r^T r
        dir_norm_sq_L  = direction @ L_direction

        for k in range(n):
            if dir_norm_sq_L <= 0:
                if k == 0:
                    return b
                else:
                    return p_star
            # Compute the new guess for the solution
            alpha    = res_norm_sq / dir_norm_sq_L
            p_star   = p_star + direction * alpha
            residual = residual + L_direction * alpha

            # Check that the new residual norm is small enough
            new_res_norm_sq = torch.sum(residual ** 2, dim=0)
            if torch.sqrt(new_res_norm_sq) < min(0.5, torch.sqrt(grad_norm))*grad_norm:
                    break

            # Update quantities
            beta          = new_res_norm_sq / res_norm_sq
            direction     = - residual + direction * beta
            L_direction   = L_method(direction)
            res_norm_sq   = new_res_norm_sq
            dir_norm_sq_L = direction@L_direction
        
        # print("Num conjugate gradient steps: " + str(k))
        return p_star

def linear_solve(L_method, b):
    '''
    Solves Ax = b where A is positive definite.
    A has shape (n, n), x and b have shape (n,).

    Input:
    - L_method : a method that takes x and returns the product Ax
    - b        : right hand side of the linear system
    Output:
    - x_star : np array of shape (n,) solving the linear system
    '''
    dim = b.shape[0]
    def LHSnp(x):
        return to_numpy(L_method(torch.tensor(x)))
    LHS_op = LinearOperator((dim, dim), matvec=LHSnp)
    x_star_np, _  = cg(LHS_op, to_numpy(b))
    x_star = torch.tensor(x_star_np)
    return x_star

def compute_inverse_approximate_hessian_matrix(sk, yk, invB_prev):
    '''
    Input:
    - sk        : previous step x_{k+1} - x_k, shape (n, 1)
    - yk        : grad(f)_{k+1} - grad(f)_{k}, shape (n, 1)
    - invB_prev : previous Hessian estimate Bk, shape (n, n)
    
    Output:
    - invB_new : previous Hessian estimate Bk, shape (n, n)
    '''
    invB_new  = invB_prev.clone()
    invB_new += (sk.T @ yk + yk.T @ invB_prev @ yk) / ((sk.T @ yk) ** 2) * (sk @ sk.T)
    prod      = (invB_prev @ yk) @ sk.T
    invB_new -= (prod + prod.T) / (sk.T @ yk)
    return invB_new


def equilibrium_convergence_report_NCG(solid, v_init, n_steps, thresh=1e-3):
    '''
    Finds the equilibrium by minimizing the total energy using Newton CG.

    Input:
    - solid     : an elastic solid to optimize
    - v_init    : the initial guess for the equilibrium position
    - n_step    : number of optimization steps
    - thresh    : threshold to stop the optimization process on the gradient's magnitude

    Ouput:
    - report : a dictionary containing various quantities of interest
    '''

    solid.update_def_shape(v_init)

    energies_el  = np.zeros(shape=(n_steps+1,))
    energies_ext = np.zeros(shape=(n_steps+1,))
    residuals    = np.zeros(shape=(n_steps+1,))
    times        = np.zeros(shape=(n_steps+1,))
    energies_el[0]  = solid.energy_el
    energies_ext[0] = solid.energy_ext
    residuals[0]    = torch.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
    idx_stop        = n_steps

    t_start = time.time()
    for i in range(n_steps):
        # Take a Newton step
        solid.equilibrium_step()

        # Measure the force residuals
        energies_el[i+1]  = solid.energy_el
        energies_ext[i+1] = solid.energy_ext
        residuals[i+1]    = torch.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
        
        if residuals[i+1] < thresh:
            residuals[i+1:]    = residuals[i+1]
            energies_el[i+1:]  = energies_el[i+1]
            energies_ext[i+1:] = energies_ext[i+1]
            idx_stop = i
            break
            
        times[i+1] = time.time() - t_start
            
    report = {}
    report['energies_el']  = energies_el
    report['energies_ext'] = energies_ext
    report['residuals']    = residuals
    report['times']        = times
    report['idx_stop']     = idx_stop

    return report

def fd_validation_ext(solid):
    epsilons = torch.logspace(-9, -3, 100)
    perturb_global = 1e-3 * (2. * torch.rand(size=solid.v_def.shape) - 1.)
    solid.displace(perturb_global)
    v_def = solid.v_def.clone()
    perturb = 2. * torch.rand(size=solid.v_def.shape) - 1.
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        grad = torch.zeros(solid.f_ext.shape)
        grad[solid.free_idx] = -solid.f_ext[solid.free_idx]
        an_delta_E = (grad*perturb).sum()

        # One step forward
        solid.displace(perturb * eps)
        E1 = solid.energy_ext.clone()

        # Two steps backward
        solid.displace(-2*perturb * eps)
        E2 = solid.energy_ext.clone()

        # Compute error
        fd_delta_E = (E1 - E2)/(2*eps)    
        errors.append(abs(fd_delta_E - an_delta_E)/abs(an_delta_E))
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_elastic(solid):
    epsilons = torch.logspace(-9, -3, 100)
    perturb_global = 1e-3 * (2. * torch.rand(size=solid.v_def.shape) - 1.)
    solid.displace(perturb_global)
    
    v_def = solid.v_def.clone()
    solid.make_elastic_forces()
    
    perturb = 2. * torch.rand(size=solid.v_def.shape) - 1.
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        solid.make_elastic_forces()
        grad = torch.zeros(solid.f.shape)
        grad[solid.free_idx] = -solid.f[solid.free_idx]
        an_delta_E = (grad*perturb).sum()
        
        # One step forward
        solid.displace(perturb * eps)
        E1 = solid.energy_el.clone()

        # Two steps backward
        solid.displace(-2*perturb * eps)
        E2 = solid.energy_el.clone()

        # Compute error
        fd_delta_E = (E1 - E2)/(2*eps)    
        errors.append(abs(fd_delta_E - an_delta_E)/abs(an_delta_E))

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_elastic_differentials(solid):
    epsilons = torch.logspace(-9, 3, 500)
    perturb_global = 1e-3 * (2. * torch.rand(size=solid.v_def.shape) - 1.)
    solid.displace(perturb_global)
    
    v_def = solid.v_def.clone()
    
    perturb = 2. * torch.rand(size=solid.v_def.shape) - 1.
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        perturb_0s = torch.zeros_like(perturb)
        perturb_0s[solid.free_idx] = perturb[solid.free_idx]
        an_df      = solid.compute_force_differentials(perturb_0s)[solid.free_idx, :]
        an_df_full = torch.zeros(solid.f.shape)
        an_df_full[solid.free_idx] = an_df.clone()
        
        # One step forward
        solid.displace(perturb * eps)
        f1 = solid.f[solid.free_idx, :]
        f1_full = torch.zeros(solid.f.shape)
        f1_full[solid.free_idx] = f1

        # Two steps backward
        solid.displace(-2*perturb * eps)
        f2 = solid.f[solid.free_idx, :]
        f2_full = torch.zeros(solid.f.shape)
        f2_full[solid.free_idx] = f2

        # Compute error
        fd_delta_f = (f1_full - f2_full)/(2*eps)   
        norm_an_df = torch.linalg.norm(an_df_full)
        norm_error = torch.linalg.norm(an_df_full - fd_delta_f)
        errors.append(norm_error/norm_an_df)

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()
