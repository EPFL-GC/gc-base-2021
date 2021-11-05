import numpy as np
import time

def conjugate_gradient(L_method, b):
        '''
        Finds an inexact Newton descent direction using Conjugate Gradient (CG)
        Solves partially L(x) = b where A is positive definite, using CG.
        The method should be take care of checking whether the added direction is
        an ascent direction or not, and whether the residuals are small enough.
        Details can be found in the handout.

        Input:
        - L_method : a method that computes the Hessian vector product. It should
                     take an array of shape (n,) and return an array of shape (n,)
        - b        : right hand side of the linear system (n,)

        Output:
        - p_star : np array of shape (n,) solving the linear system approximately
        '''

        n = b.shape[0]
        
        return np.zeros_like(b)


def compute_inverse_approximate_hessian_matrix(sk, yk, invB_prev):
    '''
    Input:
    - sk        : previous step x_{k+1} - x_k, shape (n, 1)
    - yk        : grad(f)_{k+1} - grad(f)_{k}, shape (n, 1)
    - invB_prev : previous Hessian estimate Bk, shape (n, n)
    
    Output:
    - invB_new : previous Hessian estimate Bk, shape (n, n)
    '''
    invB_new  = invB_prev.copy()
    invB_new += (sk.T @ yk + yk.T @ invB_prev @ yk) / ((sk.T @ yk) ** 2) * (sk @ sk.T)
    prod      = (invB_prev @ yk) @ sk.T
    invB_new -= (prod + prod.T) / (sk.T @ yk)
    return invB_new


def equilibrium_convergence_report_GD(solid, v_init, n_steps, step_size, thresh=1e-3):
    '''
    Finds the equilibrium by minimizing the total energy using gradient descent.

    Input:
    - solid     : an elastic solid to optimize
    - v_init    : the initial guess for the equilibrium position
    - n_step    : number of optimization steps
    - step_size : scaling factor of the gradient when taking the step
    - thresh    : threshold to stop the optimization process on the gradient's magnitude

    Ouput:
    - report : a dictionary containing various quantities of interest
    '''

    solid.update_def_shape(v_init)

    energies_el  = np.zeros(shape=(n_steps+1,))
    energies_ext = np.zeros(shape=(n_steps+1,))
    residuals    = np.zeros(shape=(n_steps+1,))
    times        = np.zeros(shape=(n_steps+1,))
    step_sizes   = np.zeros(shape=(n_steps,))

    energies_el[0]  = solid.energy_el
    energies_ext[0] = solid.energy_ext
    residuals[0]    = np.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
    idx_stop        = n_steps

    energy_tot_prev = energies_el[0] + energies_ext[0]

    t_start = time.time()
    for i in range(n_steps):
        ## TODO: Find the descent direction
        ## descent_dir has shape (#v, 3)
        descent_dir = ...

        step_size_tmp  = step_size
        max_l_iter = 20
        for l_iter in range(max_l_iter):
            step_size_tmp   *= 0.5
            solid.displace(step_size_tmp * descent_dir)

            ## TODO: Check if the armijo rule is satisfied
            ## energy_tot_tmp is the current total energy
            ## armijo is a boolean that says whether the condition is satisfied
            energy_tot_tmp = ...
            armijo         = ...
            
            if armijo or l_iter == max_l_iter-1:
                break
            else:
                solid.displace(-step_size_tmp * descent_dir)
        step_sizes[i] = step_size_tmp

        # Measure the force residuals
        energies_el[i+1]  = solid.energy_el
        energies_ext[i+1] = solid.energy_ext
        residuals[i+1]    = np.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
        energy_tot_prev   = energy_tot_tmp
        
        if residuals[i+1] < thresh:
            energies_el[i+1:]  = energies_el[i+1]
            energies_ext[i+1:] = energies_ext[i+1]
            residuals[i+1:]    = residuals[i+1]
            idx_stop = i
            break
            
        times[i+1] = time.time() - t_start
            
    report = {}
    report['energies_el']  = energies_el
    report['energies_ext'] = energies_ext
    report['residuals']    = residuals
    report['times']        = times
    report['idx_stop']     = idx_stop
    report['step_sizes']     = step_sizes

    return report


def equilibrium_convergence_report_BFGS(solid, v_init, n_steps, step_size, thresh=1e-3):
    '''
    Finds the equilibrium by minimizing the total energy using BFGS.

    Input:
    - solid     : an elastic solid to optimize
    - v_init    : the initial guess for the equilibrium position
    - n_step    : number of optimization steps
    - step_size : scaling factor of the direction when taking the step
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
    ## TODO: Collect free vertex positions
    ## grad_tmp is the current flattened gradient of the total energy
    ##          with respect to the free vertices
    grad_tmp        = ...
    residuals[0]    = ...
    idx_stop        = ...

    energy_tot_prev = energies_el[0] + energies_ext[0]


    ## TODO: Collect free vertex positions
    ## v_tmp are the current flattened free vertices
    v_tmp     = ...
    dir_zeros = np.zeros_like(solid.v_def)
    invB_prev = np.eye(v_tmp.shape[0])

    t_start = time.time()
    for i in range(n_steps):

        dir_tmp = - invB_prev @ grad_tmp
        dir_zeros[solid.free_idx, :] = dir_tmp.reshape(-1, 3)

        step_size_tmp  = step_size
        max_l_iter = 20
        for l_iter in range(max_l_iter):
            step_size_tmp   *= 0.5
            solid.displace(step_size_tmp * dir_zeros)

            ## TODO: Check if the armijo rule is satisfied
            ## energy_tot_tmp is the current total energy
            ## armijo is a boolean that says whether the condition is satisfied
            energy_tot_tmp = ...
            armijo         = ...
            
            if armijo or l_iter == max_l_iter-1:
                break
            else:
                solid.displace(-step_size_tmp * dir_zeros)
        
        ## TODO: Update all quantities
        ## v_new are the new flattened free vertices
        ## grad_new is the new flattened gradient of the total energy
        ##          with respect to the free vertices
        v_new     = ...
        grad_new  = ...
        invB_prev = compute_inverse_approximate_hessian_matrix(v_new - v_tmp, 
                                                               grad_new - grad_tmp,
                                                               invB_prev)
        v_tmp     = v_new.copy()
        grad_tmp  = grad_new.copy()

        energies_el[i+1]  = solid.energy_el
        energies_ext[i+1] = solid.energy_ext
        residuals[i+1]    = np.linalg.norm(grad_tmp)
        
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
    residuals[0]    = np.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
    idx_stop        = n_steps

    t_start = time.time()
    for i in range(n_steps):
        # Take a Newton step
        solid.equilibrium_step()

        # Measure the force residuals
        energies_el[i+1]  = solid.energy_el
        energies_ext[i+1] = solid.energy_ext
        residuals[i+1]    = np.linalg.norm((solid.f + solid.f_ext)[solid.free_idx, :])
        
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

import matplotlib.pyplot as plt
def fd_validation_ext(solid):
    epsilons = np.logspace(-9,-3,100)
    perturb_global = np.random.uniform(-1e-3, 1e-3, size=solid.v_def.shape)
    solid.displace(perturb_global)
    v_def = solid.v_def.copy()
    perturb = np.random.uniform(-1, 1, size=solid.v_def.shape)
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        grad = np.zeros(solid.f_ext.shape)
        grad[solid.free_idx] = -solid.f_ext.copy()[solid.free_idx]
        an_delta_E = (grad*perturb).sum()

        # One step forward
        solid.displace(perturb * eps)
        E1 = solid.energy_ext.copy()

        # Two steps backward
        solid.displace(-2*perturb * eps)
        E2 = solid.energy_ext.copy()

        # Compute error
        fd_delta_E = (E1 - E2)/(2*eps)    
        errors.append(abs(fd_delta_E - an_delta_E)/abs(an_delta_E))
    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_elastic(solid):
    epsilons = np.logspace(-9,-3,100)
    perturb_global = np.random.uniform(-1e-3, 1e-3, size=solid.v_def.shape)
    solid.displace(perturb_global)
    
    v_def = solid.v_def.copy()
    solid.make_elastic_forces()
    
    perturb = np.random.uniform(-1, 1, size=solid.v_def.shape)
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        solid.make_elastic_forces()
        grad = np.zeros(solid.f.shape)
        grad[solid.free_idx] = -solid.f.copy()[solid.free_idx]
        an_delta_E = (grad*perturb).sum()
        
        # One step forward
        solid.displace(perturb * eps)
        E1 = solid.energy_el.copy()

        # Two steps backward
        solid.displace(-2*perturb * eps)
        E2 = solid.energy_el.copy()

        # Compute error
        fd_delta_E = (E1 - E2)/(2*eps)    
        errors.append(abs(fd_delta_E - an_delta_E)/abs(an_delta_E))
        solid.displace(perturb * eps)

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()

def fd_validation_elastic_differentials(solid):
    epsilons = np.logspace(-9, 3,500)
    perturb_global = 1e-3*np.random.uniform(-1., 1., size=solid.v_def.shape)
    solid.displace(perturb_global)
    
    v_def = solid.v_def.copy()
    
    perturb = np.random.uniform(-1, 1, size=solid.v_def.shape)
    errors = []
    for eps in epsilons:
        # Back to original
        solid.update_def_shape(v_def)
        perturb_0s = np.zeros_like(perturb)
        perturb_0s[solid.free_idx] = perturb[solid.free_idx]
        an_df      = solid.compute_force_differentials(perturb_0s)[solid.free_idx, :]
        an_df_full = np.zeros(solid.f.shape)
        an_df_full[solid.free_idx] = an_df.copy()
        
        # One step forward
        solid.displace(perturb * eps)
        f1 = solid.f[solid.free_idx, :]
        f1_full = np.zeros(solid.f.shape)
        f1_full[solid.free_idx] = f1

        # Two steps backward
        solid.displace(-2*perturb * eps)
        f2 = solid.f[solid.free_idx, :]
        f2_full = np.zeros(solid.f.shape)
        f2_full[solid.free_idx] = f2

        # Compute error
        fd_delta_f = (f1_full - f2_full)/(2*eps)   
        norm_an_df = np.linalg.norm(an_df_full)
        norm_error = np.linalg.norm(an_df_full - fd_delta_f)
        errors.append(norm_error/norm_an_df)

    plt.loglog(epsilons, errors)
    plt.grid()
    plt.show()