
import numpy as np


# -----------------------------------------------------------------------------
#  UTILITIES
# -----------------------------------------------------------------------------

def uv_grid(u_domain, v_domain, n_u, n_v):
    """Computes a triangulation of a u-v patch.

    Parameters:
    - u_domain : list of floats [u_min, u_max]
        The u domain.
    - v_domain : list of floats [v_min, v_max]
        The v domain.
    - n_u : int
        The number of samplings in u direction.
    - n_v : int
        The number of samplings in v direction.

    Returns:
    - P : np.array (n_u * n_v, 2)
        The sample points. The i-th row contains the (u, v) coordinates of
        point p_i.
    - F : np.array (2 * (n_u - 1) * (n_v - 1), 3)
        Triangular mesh faces of the points P. This can be used for
        visualization.
    """
    G = np.arange(n_u * n_v).reshape(n_v, n_u)
    v1 = G[:-1, :-1].flatten()
    v2 = G[:-1, 1:].flatten()
    v3 = G[1:, 1:].flatten()
    v4 = G[1:, :-1].flatten()
    F1 = np.column_stack((v1, v3, v4))
    F2 = np.column_stack((v1, v2, v3))
    F = np.vstack((F1, F2))
    u, v = np.meshgrid(np.linspace(u_domain[0], u_domain[1], n_u),
                       np.linspace(v_domain[0], v_domain[1], n_v))
    P = np.column_stack((u.flatten(), v.flatten()))
    return P, F


def bgr_color(values):
    """Computes bgr colors of some values.

    Parameters:
    - values * np.array (#values, )

    Returns:
    - rgb : np.array(#values, 3)
    """
    val = np.array(values)
    expansion = .5
    X = np.linspace(0, 1, 64)
    increase = 255 * (X ** expansion)
    decrease = increase[::-1]
    one = 255 * np.ones(64)
    zero = np.zeros(64)
    lut = np.ones((256, 3)).astype(float)
    B = np.hstack((one, decrease, zero, zero))
    G = np.hstack((increase, one, one, decrease))
    R = np.hstack((zero, zero, increase, one))
    lut[:, 0] = R / 255
    lut[:, 1] = G / 255
    lut[:, 2] = B / 255
    lut[-1, :] = [255, 0, 255]
    max_val = np.max(np.abs(values))
    val = np.round(val / max_val * 127).astype('i') + 127
    return lut[val]
