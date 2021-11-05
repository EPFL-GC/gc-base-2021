import numpy as np
from objectives import *
from Utils import *

def stretch_from_point(v, stretches, v_idx):
    '''
    Input:
    - v         : vertices to be deformed (#v, 3)
    - stretches : np array of shape (3,) giving the stretching coefficients along x, y, z
    - v_idx     : vertex index around which the mesh is stretched
    
    Output:
    - v_stretched : deformed vertices (#v, 3)
    '''
    v_stretched = v
    return v_stretched

def report_stretches(solid, v_rest_init, bv, v_target, stretches, lowest_pin_idx):
    '''
    Input:
    - solid          : an elastic solid to deform
    - v_rest_init    : reference vertex position to compress/stretch (#v, 3)
    - bv             : boundary vertices index (#bv,)
    - v_target       : target boundary vertices position
    - stretches      : np array of shape (n_stretches,) containing a stretch factors to try out
    - lowest_pin_idx : index of the pinned index that has the lowest z coordinate
    
    Output:
    - list_v_rest      : list of n_stretches rest vertex positions that have been tried out
    - list_v_eq        : list of the corresponding n_stretches equilibrium vertex positions
    - target_closeness : np array of shape (n_stretches,) containing the objective values for each stretch factor
    '''
    list_v_rest = []
    list_v_eq   = []
    target_closeness = np.zeros(shape=(stretches.shape[0],))
    v_rest_tmp  = v_rest_init.copy()

    for i, stretch in enumerate(stretches):
        # Compute and update the rest shape of the solid

        # Compute new equilibrium (use the previous equilibrium state as initial guess if available)
        # You may use equilibrium_convergence_report_NCG to find the equilibrium (20 steps and thresh=1 should do)
        
        # Update guess for next stretch factor

        # Fill in the 
        list_v_rest.append(solid.v_rest.copy())
        list_v_eq.append(solid.v_def.copy())
        target_closeness[i] = objective_target_BV(solid.v_def, v_target, bv)
    
    return list_v_rest, list_v_eq, target_closeness