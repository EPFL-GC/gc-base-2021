import numpy as np

from scipy import sparse

from scipy.sparse import linalg

import igl

from smooth_surfaces import *

from utils import *


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def compute_orthogonal_frames(N):
    """Computes an orthonormal frame {e1, e2, e3} at vertices x_i.

    Parameters:
    - N : np.array (|n|, 3)
        The i-th row contains a vector in direction e3 at x_i.

    Returns:
    - e1: np.array (|n|, 3)
        The i-th row contains the axis e1 at vertex x_i.
    - e2: np.array (|n|, 3)
        The i-th row contains the axis e2 at vertex x_i.
    - e3: np.array (|n|, 3)
        The i-th row contains the axis e3 at vertex x_i.
    """
    e3 = N / np.linalg.norm(N, axis=1, keepdims=True)
    e1 = np.zeros(e3.shape)
    e1[:, 0] = - e3[:, 1]
    e1[:, 1] = e3[:, 0]
    e1[np.where((e1[:, 0] == 0) & (e1[:, 1] == 0))[0], 1] = 1
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def vertex_double_rings(F):
    """Computes double rings of mesh vertices.

    Parameters:
    - F : np.array (|F|, 3)
        The array of triangle faces.

    Returns:
    - v_i : np.array (n, )
        The indices of the central vertices i.
    - v_j : np.array (n, )
        The indices of the vertices j connected to vertex i by at most two
        edges, such that v_j[k] belongs to the double ring of v_i[k].
    """
    M = igl.adjacency_matrix(F)
    vi, vj = M.nonzero()
    N = M[vj]
    vii, vjj = N.nonzero()
    L = sparse.coo_matrix((np.ones(len(vii)), (vii, np.arange(len(vii)))))
    k = np.array(L.sum(axis=1)).flatten().astype('i')
    vii = np.repeat(vi, k)
    M = sparse.coo_matrix((np.ones(len(vii)), (vii, vjj)), shape=M.shape)
    M = M.tolil()
    M.setdiag(0)
    return M.nonzero()


# -----------------------------------------------------------------------------
# OSCULATING PARABOLOID
# -----------------------------------------------------------------------------

def compute_osculating_paraboloids(V, F, e1, e2, e3):
    """Computes the coefficients of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y
    through least squares fitting. Try to vectorize this function.

    Parameters:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the global coordinates of the vertex x_i in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.
    - e1: np.array (|V|, 3)
        The i-th row contains the axis e1 at vertex x_i.
    - e2: np.array (|V|, 3)
        The i-th row contains the axis e2 at vertex x_i.
    - e3: np.array (|V|, 3)
        The i-th row contains the axis e3 at vertex x_i.

    Returns:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.
    """
    # we compute the indices for the double ring vj at each vertex vi
    vi, vj = vertex_double_rings(F)
    # TODO: fill the function
    a = None
    return a


def compute_osculating_paraboloid_first_derivatives(a):
    """Computes the first derivatives of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y,
    evaluated at the point x_i. Try to vectorize this function.

    Parameters:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.

    Returns:
    - x_x : np.array (|V|, 3)
        The first derivatives x_x, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_x(x_i).
    - x_y : np.array (|V|, 3)
        The second derivatives x_y, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_y(x_i).
    """
    # TODO: fill the function
    x_x = None
    x_y = None
    return x_x, x_y


def compute_osculating_paraboloid_second_derivatives(a):
    """Computes the second derivatives of the osculating paraboloid at vertices x_i
    in local orthonormal coordinates with base {x_i; e1, e2, e3}, with
    eta(x,y) = a x^2 + b y^2 + c xy + d x + e y,
    evaluated at the point x_i. Try to vectorize this function.

    Parameters:
    - a : np.array (|V|, 5)
        The paraboloid coefficients. i-th row contains the coefficients
        [a, b, c, d, e] of the paraboloid at x_i.

    Returns:
    - x_xx : np.array (|V|, 3)
        The second derivatives x_xx, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_xx(x_i).
    - x_xy : np.array (|V|, 3)
        The second derivatives x_xy, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_xy(x_i).
    - x_yy : np.array (|V|, 3)
        The second derivatives x_yy, where the i-th row contains the local (x,y,z)
        coordinates of the vector x_yy(x_i).
    """
    # TODO: fill the function
    x_xx = None
    x_xy = None
    x_yy = None
    return x_xx, x_xy, x_yy


def compute_mesh_principal_curvatures(V, F):
    """Computes the principal curvatures at mesh vertices v_i through quadratic
    fitting.

    Parameters:
    - V : np.array (|V|, 3)
        The array of vertices positions.
        Contains the global coordinates of the vertex x_i in i-th row
    - F : np.array (|F|, 3)
        The array of triangle faces.

    Returns:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature
        at vertex x_i.
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature
        at vertex x_i.
    - d_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the global coordinates of d_1(x_i).
    - d_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the global coordinates of d_2(x_i).
    """
    # we compute a vertex normal with libigl and use it as local axis e3:
    N = igl.per_vertex_normals(V, F)
    # then we compute the local axes:
    e1, e2, e3 = compute_orthogonal_frames(N)
    # TODO: fill the function. Use the functions implemented in assignment 3.1
    k_1 = None
    k_2 = None
    d_1 = None
    d_2 = None
    return k_1, k_2, d_1, d_2
