import torch
from utils import linear_solve

torch.set_default_dtype(torch.float64)

def compute_adjoint(solid ,dJdx_U):
    '''
    This assumes that S is at equilibrium when called
    
    Input:
    - solid : an elastic solid at equilibrium
    - dJdx_U  : array of shape (3*#unpinned,)
    
    Output:
    - adjoint : array of shape (3*#v,)
    '''
        
    def LHS(dx):
        '''
        Should implement the Hessian-Vector product (taking into account pinning constraints) as described in the handout. 
        '''
        return None
    
    RHS = ...
    
    _ = linear_solve(LHS, RHS)

    return None