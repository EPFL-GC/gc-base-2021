import numpy as np
from geometry import compute_mesh_centroid
# -----------------------------------------------------------------------------
#                               Mesh geometry
# -----------------------------------------------------------------------------

def shear_transformation(V, nu):
    """
    Computes vertices' postion after the shear transformation.

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - nu : the shear paramter
    
    Output:
    - V1 : np.array (|V|, 3)
        The array of vertices positions after transformation.
    """
    V1 = V.copy()


    # HW1 1.3.6
    # enter your code here

    return V1


def shear_equilibrium(V, F, x_csl):
    """
    Shear the input mesh to make it equilibrium.

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    - x_csl: np.array (3, )
        The x coordinate of the target centroid
    Output:
    - V1 : np.array (|V|, 3)
        The array of vertices positions that are equilibrium.
    """
    V1 = V.copy()

    # HW1 1.3.7
    # enter your code here

    return V1
