from elasticenergy import *
from elasticsolid import *
import torch

class ObjectiveBV:
    def __init__(self, vt_surf, bv):
        self.vt_surf, self.bv= vt_surf, bv
    
    def obj(self, v):
        return objective_target_BV(v, self.vt_surf, self.bv)
    
    def grad(self, v):
        return grad_objective_target_BV(v, self.vt_surf, self.bv)

class ObjectiveReg:
    def __init__(self, params_init, params_idx, harm_int, weight_reg=0, force_scale=1e7, length_scale=1.):
        self.params_init, self.params_idx, self.harm_int = params_init, params_idx, harm_int
        self.weight_reg, self.force_scale, self.length_scale = weight_reg, force_scale, length_scale
    
    def obj(self, solid, params_tmp):
        if self.weight_reg == 0: return 0
        return self.length_scale * self.weight_reg / self.force_scale *regularization_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)
    
    def grad(self, solid, params_tmp):
        if self.weight_reg == 0: return torch.zeros_like(self.params_init)
        return self.length_scale * self.weight_reg / self.force_scale *regularization_grad_neo_hookean(self.params_init, solid, params_tmp, self.params_idx, self.harm_int)


def objective_target_BV(v, vt, bv):
    '''
    Input:
    - v  : array of shape (#v, 3), containing the current vertices position
    - vt : array of shape (#bv, 3), containing the target surface 
    - bv : boundary vertices index (#bv,)
    
    Output:
    - objective : single scalar measuring the deviation from the target shape
    '''
    return None

def grad_objective_target_BV(v, vt, bv):
    '''
    Input:
    - v  : array of shape (#v, 3), containing the current vertices position
    - vt : array of shape (#bv, 3), containing the target surface 
    - bv : boundary vertices (#bv,)
    
    Output:
    - gradient : array of shape (#v, 3)
    '''
    return None

def regularization_neo_hookean(params_prev, solid, params, params_idx, harm_int):
    '''
    Input:
    - params_prev : array of shape (3*#params,) containing the previous shape
    - solid       : an elastic solid to copy
    - params      : array of shape (3*#params,) containing the new shape parameters
    - params_idx  : parameters index in the vertex list. Has shape (#params,)
    - harm_int    : an harmonic interpolator
    
    Output:
    - energy    : the neo hookean energy
    '''
    
    v_prev = solid.v_rest.detach()
    v_prev[params_idx, :] = params_prev.reshape(-1, 3)
    v_prev = harm_int.interpolate_fill(v_prev)

    f_mass = torch.zeros(size=(3,))

    ee_tmp        = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
    solid_virtual = ElasticSolid(v_prev, solid.t, ee_tmp, rho=solid.rho, 
                                 pin_idx=solid.pin_idx, f_mass=f_mass)
    
    v_new = solid_virtual.v_rest.detach()
    v_new[params_idx, :] = params.reshape(-1, 3)
    v_new = harm_int.interpolate_fill(v_new)
    solid_virtual.update_def_shape(v_new)

    solid_virtual.make_elastic_energy()
    
    return solid_virtual.energy_el

def regularization_grad_neo_hookean(params_prev, solid, params, params_idx, harm_int):
    '''
    Input:
    - params_prev : array of shape (3*#params,) containing the previous shape
    - solid       : an elastic solid to copy
    - params      : array of shape (3*#params,) containing the new shape parameters
    - params_idx  : parameters index in the vertex list. Has shape (#params,)
    - harm_int    : an harmonic interpolator
    
    Output:
    - grad_reg    : array of shape (3*#params,), the regularization gradient
    '''

    v_prev = solid.v_rest.detach()
    v_prev[params_idx, :] = params_prev.reshape(-1, 3)
    v_prev = harm_int.interpolate_fill(v_prev)

    ee_tmp        = NeoHookeanElasticEnergy(solid.ee.young, solid.ee.poisson)
    solid_virtual = ElasticSolid(v_prev, solid.t, ee_tmp, rho=solid.rho, 
                                 pin_idx=solid.pin_idx, f_mass=solid.f_mass)
    
    v_new = solid_virtual.v_rest.detach()
    v_new[params_idx, :] = params.reshape(-1, 3)
    v_new = harm_int.interpolate_fill(v_new)
    solid_virtual.update_def_shape(v_new)
    
    # Negative of the elastic forces: gradient of the energy
    grad_reg = - solid_virtual.f[params_idx].reshape(-1,)
    return grad_reg
