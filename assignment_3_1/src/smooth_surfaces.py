
import numpy as np


# -----------------------------------------------------------------------------
#  DERIVATIVES OF PARABOLOID
# -----------------------------------------------------------------------------

def compute_paraboloid_points(P, a, b, c, d, e):
    """Computes the points of the paraboloid x(u,v) = (u, v, z(u,v)) with
    z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x : np.array (n, 3)
        The points x(P), where the i-th row contains the (x,y,z) coordinates
        of the point x(p_i)
    """
    x = None
    return x


def compute_paraboloid_first_derivatives(P, a, b, c, d, e):
    """Computes the first derivatives of the paraboloid x(u,v) = (u, v, z(u,v))
    with z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x_u : np.array (n, 3)
        The vectors x_u(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The vectors x_v(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_v(p_i).
    """
    x_u = None
    x_v = None
    return x_u, x_v


def compute_paraboloid_second_derivatives(P, a, b, c, d, e):
    """Computes the second derivatives of the paraboloid x(u,v) = (u, v, z(u,v))
    with z(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - a, b, c, d, e : float
        The parameters of the paraboloid.

    Returns:
    - x_uu : np.array (n, 3)
        The vectors x_uu(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
        The vectors x_uv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
        The vectors x_vv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_vv(p_i).
    """
    x_uu = None
    x_uv = None
    x_vv = None
    return x_uu, x_uv, x_vv


# -----------------------------------------------------------------------------
#  DERIVATIVES OF TORUS
# -----------------------------------------------------------------------------

def compute_torus_points(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.
    Returns:
    - x : np.array (n, 3)
        The points x(P), where the i-th row contains the (x,y,z) coordinates
        of the point x(p_i)
    """
    x = None
    return x


def compute_torus_first_derivatives(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.
    Returns:
    - x_u : np.array (n, 3)
        The vectors x_u(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The vectors x_v(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_v(p_i).
    """
    x_u = None
    x_v = None
    return x_u, x_v


def compute_torus_second_derivatives(P, R, r):
    """Computes the second derivatives of a torus.
    Try to vectorize this function.

    Parameters:
    - P : np.array (n, 2)
        Contains in i-th row the (u, v) coordinates of the i-th parameter point p_i.
    - R : float
        The radius of revolution.
    - r : float
        The radius of the cross section.

    Returns:
    - x_uu : np.array (n, 3)
        The vectors x_uu(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
        The vectors x_uv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
        The vectors x_vv(P), where the i-th row contains the (x,y,z) coordinates
        of the vector x_vv(p_i).
    """
    x_uu = None
    x_uv = None
    x_vv = None
    return x_uu, x_uv, x_vv


# -----------------------------------------------------------------------------
#  SHAPE OPERATOR
# -----------------------------------------------------------------------------

def compute_first_fundamental_form(x_u, x_v):
    """Computes the first fundamental form I.
    Try to vectorize this function.

    Parameters:
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - I : np.array (n, 2, 2)
        The first fundamental forms.
        The (i, j, k) position contains the (j, k) element of the first
        fundamental form I(p_i).
    """
    I = None
    return I


def compute_surface_normal(x_u, x_v):
    """Computes the surface normal n.
    Try to vectorize this function.

    Parameters:
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - n : np.array (n, 3)
        The surface normals.
        The i-th row contains the (x,y,z) coordinates of the vector n(p_i).
    """
    n = None
    return n


def compute_second_fundamental_form(x_uu, x_uv, x_vv, n):
    """Computes the second fundamental form II.
    Try to vectorize this function.

    Parameters:
    - x_uu : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_uu(p_i).
    - x_uv : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_uv(p_i).
    - x_vv : np.array (n, 3)
         The i-th row contains the (x,y,z) coordinates of the vector x_vv(p_i).
    - n : np.array (n, 3)
        The surface normals.
        The i-th row contains the (x,y,z) coordinates of the vector n(p_i).

    Returns:
    - II : np.array (n, 2, 2)
        The second fundamental forms.
        The (i, j, k) position contains the (j, k) element of the second
        fundamental form II(p_i).
    """
    II = None
    return II


def compute_shape_operator(I, II):
    """Computes the shape operator S.
    Try to vectorize this function.

    Parameters:
    - I : np.array (n, 2, 2)
        The first fundamental forms.
        The (i, j, k) position contains the (j, k) element of the first
        fundamental form I(p_i).
    - II : np.array (n, 2, 2)
        The second fundamental forms.
        The (i, j, k) position contains the (j, k) element of the second
        fundamental form II(p_i).

    Returns:
    - S : np.array (n, 2, 2)
        The shape operators.
        The (i, j, k) position contains the (j, k) element of the shape
        operator S(p_i).
    """
    S = None
    return S


# -----------------------------------------------------------------------------
#  PRINCIPAL CURVATURES
# -----------------------------------------------------------------------------

def compute_principal_curvatures(S, x_u, x_v):
    """Computes principal curvatures and corresponding principal directions.
    Try to vectorize this function.

    Parameters:
    - S : np.array (n, 2, 2)
        The shape operators.
        The (i, j, k) position contains the (j, k) element of the shape
        operator S(p_i).
    - x_u : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_u(p_i).
    - x_v : np.array (n, 3)
        The i-th row contains the (x,y,z) coordinates of the vector x_v(p_i).

    Returns:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature k_1(p_i).
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature k_2(p_i).
    - e_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the (x,y,z) coordinates of e_1(p_i).
    - e_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the (x,y,z) coordinates of e_2(p_i).
    """
    # this section computes the ordered eigenvalues and eigenvectors of S where
    # k_1[i] = min eigenvalue at p_i, k_2[i] = max eigenvalue at p_i,
    # bar_e_1[i] = [u, v] components of the eigenvector of k_1,
    # bar_e_2[i] = [u, v] components of the eigenvector of k_2
    eig = np.linalg.eig(S)
    index = np.argsort(eig[0], axis=1)
    k_1 = eig[0][np.arange(len(S)), index[:, 0]]
    k_2 = eig[0][np.arange(len(S)), index[:, 1]]
    bar_e_1 = eig[1][np.arange(len(S)), :, index[:, 0]]
    bar_e_2 = eig[1][np.arange(len(S)), :, index[:, 1]]

    # TODO: compute the normalized 3D vectors e_1, e_2
    e_1 = None
    e_2 = None
    return k_1, k_2, e_1, e_2


# -----------------------------------------------------------------------------
#  ASYMPTOTIC DIRECTIONS
# -----------------------------------------------------------------------------

def compute_asymptotic_directions(k_1, k_2, e_1, e_2):
    """Computes principal curvatures and corresponding principal directions.
    Try to vectorize this function.

    Parameters:
    - k_1 : np.array (n)
        The min principal curvature. i-th element contains the curvature k_1(p_i).
    - k_2 : np.array (n)
        The max principal curvature. i-th element contains the curvature k_2(p_i).
    - e_1 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_1.
        The i-th row contains the (x,y,z) coordinates of e_1(p_i).
    - e_2 : np.array (n, 3)
        The unitized principal curvature direction corresponding to k_2.
        The i-th row contains the (x,y,z) coordinates of e_2(p_i).

    Returns:
    - a_1 : np.array (n, 3)
        The first unitized asymptotic direction. The i-th row contains the
        (x,y,z) coordinates of a_2(p_i) if it exists, (0, 0, 0) otherwise.
    - a_2 : np.array (n, 3)
        The second unitized asymptotic direction. The i-th row contains the
        (x,y,z) coordinates of a_2(p_i) if it exists, (0, 0, 0) otherwise.
    """
    a_1 = None
    a_2 = None
    return a_1, a_2

