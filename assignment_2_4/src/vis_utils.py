import meshplot as mp
import numpy as np
# import jax.numpy as jnp
import torch
import torch

shadingOptions = {
    "flat":True,
    "wireframe":False,   
}

def to_numpy(tensor):
    return tensor.detach().clone().numpy()

def plot_torch_solid(solid, be, rot, length_scale, target_mesh=None):
    '''
    Input:
    - solid        : elastic solid to visualize
    - be           : boundary edges
    - rot          : transformation matrix to apply (here we assume it is a rotation)
    - length_scale : length scale of the mesh, used to represent pinned vertices
    - target_mesh  :
    '''
    p = mp.plot(to_numpy(solid.v_def) @ rot.T, to_numpy(solid.t), shading=shadingOptions)
    p.add_points(to_numpy(solid.v_def[solid.pin_idx, :]) @ rot.T, shading={"point_color":"black", "point_size": 0.1 * length_scale})
    forcesScale = 2 * torch.max(torch.linalg.norm(solid.f_ext, axis=1))
    p.add_lines(to_numpy(solid.v_def) @ rot.T, to_numpy(solid.v_def + solid.f_ext / forcesScale) @ rot.T)
    p.add_edges(to_numpy(solid.v_rest) @ rot.T, be, shading={"line_color": "blue"})
    if not target_mesh is None:
        p.add_edges(to_numpy(target_mesh) @ rot.T, be, shading={"line_color": "red"})