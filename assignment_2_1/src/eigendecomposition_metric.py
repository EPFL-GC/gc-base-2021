import numpy as np
import meshplot as mp
import igl

def compute_eigendecomposition_metric(jac):
    '''
    Input:
    - jac : a np array of shape (#t, 3, 3) containing the stacked jacobians

    Output:
    - squared_eigvals : np array of shape (#t, 3) containing the square root of the eigenvalues of the metric tensor
    - eigvecs         : np array of shape (#t, 3, 3) containing the eigenvectors of the metric tensor
    '''

    eigvals, eigvecs = np.zeros(shape=jac.shape[:2]), np.zeros_like(jac)

    return eigvals, eigvecs

def plot_eigendecomposition_metric(solid, squared_eigvals, eigvecs, rot, scale=0.05):
    '''
    Input:
    - solid           : an ElasticSolid object
    - squared_eigvals : np array of shape (#t, 3) containing the square root of the eigenvalues of the metric tensor
    - eigvecs         : np array of shape (#t, 3, 3) containing the eigenvectors of the metric tensor
    - rot             : a rotation matrix for plotting purposes
    - scale           : scaling for plotting purposes
    '''

    scaled_eigvecs = scale * np.einsum('ik, ijk -> ijk', squared_eigvals, eigvecs)

    start_plot0 = (solid.def_barycenters - scaled_eigvecs[..., 0]) @ rot.T
    start_plot1 = (solid.def_barycenters - scaled_eigvecs[..., 1]) @ rot.T
    start_plot2 = (solid.def_barycenters - scaled_eigvecs[..., 2]) @ rot.T
    end_plot0   = (solid.def_barycenters + scaled_eigvecs[..., 0]) @ rot.T
    end_plot1   = (solid.def_barycenters + scaled_eigvecs[..., 1]) @ rot.T
    end_plot2   = (solid.def_barycenters + scaled_eigvecs[..., 2]) @ rot.T

    # Get boundary edges
    be = igl.edges(igl.boundary_facets(solid.t))

    p = mp.plot(solid.v_def @ rot.T, be, shading={"line_color": "black"})
    p.add_points(solid.def_barycenters @ rot.T, shading={"point_color":"black", "point_size": 0.2})
    
    # In tension
    tens0 = np.argwhere(squared_eigvals[:, 0]>1. + 1e-6)
    tens1 = np.argwhere(squared_eigvals[:, 1]>1. + 1e-6)
    tens2 = np.argwhere(squared_eigvals[:, 2]>1. + 1e-6)
    if tens0.shape[0] != 0:
        p.add_lines(start_plot0[tens0, :], 
                    end_plot0[tens0, :], 
                    shading={"line_color": "#182C94"})
    if tens1.shape[0] != 0:
        p.add_lines(start_plot1[tens1, :], 
                    end_plot1[tens1, :], 
                    shading={"line_color": "#182C94"})
    if tens2.shape[0] != 0:
        p.add_lines(start_plot2[tens2, :], 
                    end_plot2[tens2, :], 
                    shading={"line_color": "#182C94"})

    # In compression
    comp0 = np.argwhere(squared_eigvals[:, 0]<1. - 1e-6)
    comp1 = np.argwhere(squared_eigvals[:, 1]<1. - 1e-6)
    comp2 = np.argwhere(squared_eigvals[:, 2]<1. - 1e-6)
    if comp0.shape[0] != 0:
        p.add_lines(start_plot0[comp0, :], 
                    end_plot0[comp0, :], 
                    shading={"line_color": "#892623"})
    if comp1.shape[0] != 0:
        p.add_lines(start_plot1[comp1, :], 
                    end_plot1[comp1, :], 
                    shading={"line_color": "#892623"})
    if comp2.shape[0] != 0:
        p.add_lines(start_plot2[comp2, :], 
                    end_plot2[comp2, :], 
                    shading={"line_color": "#892623"})

    # Neutral
    # In compression
    neut0 = np.argwhere(abs(squared_eigvals[:, 0]-1.) < 1e-6)
    neut1 = np.argwhere(abs(squared_eigvals[:, 1]-1.) < 1e-6)
    neut2 = np.argwhere(abs(squared_eigvals[:, 2]-1.) < 1e-6)
    if neut0.shape[0] != 0:
        p.add_lines(start_plot0[neut0, :], 
                    end_plot0[neut0, :], 
                    shading={"line_color": "#027337"})
    if neut1.shape[0] != 0:
        p.add_lines(start_plot1[neut1, :], 
                    end_plot1[neut1, :], 
                    shading={"line_color": "#027337"})
    if neut2.shape[0] != 0:
        p.add_lines(start_plot2[neut2, :], 
                    end_plot2[neut2, :], 
                    shading={"line_color": "#027337"})