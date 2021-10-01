import numpy as np
# -----------------------------------------------------------------------------
#                               Mesh geometry
# -----------------------------------------------------------------------------

def compute_faces_area(V, F):
    """
    Computes the area of the faces of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - area : np.array (|F|,)
        The area of the faces. The i-th position contains the area of the i-th
        face.
    """
    area = np.zeros(F.shape[0])
    
    # HW1 1.3.3
    # enter your code here

    return area


def compute_mesh_area(V, F):
    """
    Computes the area of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - area : float
        The area of the mesh.
    """
    area = 0

    # HW1 1.3.3 
    # enter your code here

    return area


def compute_faces_centroid(V, F):
    """
    Computes the area centroid of each face of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - cf : np.array (|F|, 3)
        The area centroid of the faces.
    """
    cf = np.zeros((F.shape[0], 3))

    # HW1 1.3.4
    # enter your code here

    return cf


def compute_mesh_centroid(V, F):
    """
    Computes the area centroid of a given triangle mesh (V, F).

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    Output:
    - centroid : np.array (3,)
        The area centroid of the mesh.
    """
    mc = np.zeros(3)

    # HW1 1.3.4
    # enter your code here

    return mc

def compute_center_support_line(V):
    """
    Computes the x coordinate of the center of the support line

    Input:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row

    Output:
    - x_csl : float
        the x coordinate of the center of the support line
    """
    x_csl = 0

    # HW1 1.3.5
    # enter your code here

    return x_csl
