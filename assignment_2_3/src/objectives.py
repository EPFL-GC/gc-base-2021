import numpy as np

def objective_target_BV(v, vt, bv):
    '''
    Input:
    - v  : array of shape (#v, 3), containing the current vertices position
    - vt : array of shape (#bv, 3), containing the target surface 
    - bv : boundary vertices index (#bv,)
    
    Output:
    - objective : single scalar measuring the deviation from the target shape
    '''
    return 0.