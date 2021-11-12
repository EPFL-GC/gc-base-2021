import torch
import igl
import numpy as np

torch.set_default_dtype(torch.float64)

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

class HarmonicInterpolator:
    '''
    Provides a way to interpolate boundary vertices

    Attributes:
    - lii     : block of the Laplacian matrix associated to 
                interior vertices (#iv, #iv)
    - lib     : block of the Laplacian matrix associated to 
                interior/boundary vertices (#iv, #v - #iv)
    - lii_inv : inverse of lii (#iv, #iv)
    '''

    def __init__(self, v, t, iv):
        '''
        Input:
        - v  : tensor of shape (#v, 3)
        - t  : tensor of shape (#t, 4)
        - iv : np array containing interior vertices (#iv,)
        '''
        self.lii = None
        self.lib = None
        self.lii_inv = None

        self.iv = None
        self.bv = None

        self.update_interpolator(v, t, iv)

    def update_interpolator(self, v, t, iv):
        '''
        Input:
        - v  : tensor of shape (#v, 3)
        - t  : tensor of shape (#t, 4)
        - iv : np array containing interior vertices (#iv,)
        '''

        vNP = to_numpy(v)
        tNP = to_numpy(t)
        
        self.iv = iv
        v_idx   = np.arange(v.shape[0])
        self.bv = v_idx[np.invert(np.in1d(v_idx, iv))]

        l = igl.cotmatrix(vNP, tNP).todense()
        self.lii = torch.tensor(l[iv[:, np.newaxis], self.iv])
        self.lib = torch.tensor(l[iv[:, np.newaxis], self.bv])
        self.lii_inv = torch.linalg.inv(self.lii)

    def interpolate(self, v_surf):
        '''
        Input:
        - v_surf : tensor of shape (#v - #iv, 3) for which the interior vertices 
                   should be interpolated

        Output:
        - v_inter : tensor of shape (#v, 3) for which the interior vertices 
                    have been interpolated from vertices of v_surf
        '''
        n_v = self.bv.shape[0] + self.iv.shape[0]
        v_inter = torch.zeros(size=(n_v, 3))
        v_inter[self.bv, :] = v_surf
        v_inter[self.iv, :] = - self.lii_inv @ self.lib @ v_surf
        return v_inter

    def interpolate_fill(self, v_vol):
        '''
        Input:
        - v_vol : tensor of shape (#v, 3) for which the interior vertices 
                  should be interpolated

        Output:
        - v_inter : tensor of shape (#v, 3) for which the interior vertices 
                    have been interpolated from vertices of v_surf
        '''
        n_v = self.bv.shape[0] + self.iv.shape[0]
        assert n_v == v_vol.shape[0]
        v_inter = v_vol
        v_inter[self.iv, :] = - self.lii_inv @ self.lib @ v_vol[self.bv, :]
        return v_inter
