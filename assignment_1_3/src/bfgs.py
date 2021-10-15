from energies import *
from optimization import compute_optimization_objective, compute_optimization_objective_gradient
from linesearch import *
import numpy as np
import time
import igl

def compute_approximate_hessian_matrix(sk, yk, B_prev):
    """
        Compute the approximated hessian matrix

        Input:
        - s_k : np.array(#V * 2, 1)
            s_k = x_{k+1} - x_{k}, the difference in variables at two consecutive iterations.
            Note that x_{k} is the vertices' coordinate V at the k-th iteration.
            The vertices' cooridnate V however is a np.array(#V, 3)
            We use x_{k} = V[:, 0 : 2].flatten() to flatten the vertices coordinartes.
        
        - y_k : np.array(#V * 2, 1)
            y_k = grad(f(x_{k+1})) - grad(f(x_{k})), 
            the difference in function gradient at two consecutive iterations.
            The grad(f(x_{k})) is the gradient of our objective energy,
            which is a np.array(#V, 2)
            To be used for BFGS, we flatten the gradient.
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.

        - B_prev: np.array(#V * 2, #V * 2)
            The approximated hessian matrix from last iteration

        Output
        -B_new: np.array(#V * 2, #V * 2)
            The approximated hessian matrix of current iteration

    """
    B_new = B_prev.copy()
    return B_new

def compute_inverse_approximate_hessian_matrix(sk, yk, invB_prev):
    """
        Compute the inverse approximated hessian matrix

        Input:
      - s_k : np.array(#V * 2, 1)
            s_k = x_{k+1} - x_{k}, the difference in variables at two consecutive iterations.
            Note that x_{k} is the vertices' coordinate V at the k-th iteration.
            The vertices' cooridnate V however is a np.array(#V, 3)
            We use x_{k} = V[:, 0 : 2].flatten() to flatten the vertices coordinartes.
        
        - y_k : np.array(#V * 2, 1)
            y_k = grad(f(x_{k+1})) - grad(f(x_{k})), 
            the difference in function gradient at two consecutive iterations.
            The grad(f(x_{k})) is the gradient of our objective energy,
            which is a np.array(#V, 2)
            To be used for BFGS, we flatten the gradient.
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.
            
        - invB_prev: np.array(#V * 2, #V * 2)
            The inversed matrix of the approximated hessian from last iteration

        Output
        - invB_new: np.array(#V * 2, #V * 2)
            The inversed matrix of the approximated hessian at current iteration

        """

    invB_new = invB_prev.copy()
    return invB_new


def bfgs_with_line_search(V, F, x_csl, w, obj_tol, theta, beta, c, iter):

    """
    Find equilibrium shape by using BFGS method

     Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row
    - F : np.array (#F, 3)
        The array of triangle faces.
    - x_csl : float
        The x coordinate of the center of the support line.
    - w : float
        The weight for shape preservation energy.
    - obj_tol: float
        The termination condition for optimization.
        The program stop running if
        the absolute different between the objectives of two consecutive iterations is smaller than obj_tol
    - theta : float
        The initial gradient descent step size.
    - beta : float
        The backtracking ratio, alpha = beta * alpha
    - c: float
        The coefficient for armijo condition
    - iter : int
        The number of iteration for gradient descent.

    Output:
    - V1 : np.array (#V, 3)
        The optimized mesh's vertices
    - F : np.array (#F, 3)
        The array of triangle faces.
    - energy: np.array(iters, 1)
        The objective function energy curve with respect to the number of iterations.
    - running_time: float
        The tot running time of the optimization
    """

    V1 = V.copy()

    # this function of libigl returns an array (#edges, 2) where i-th row
    # contains the indices of the two vertices of i-th edge.
    E = igl.edges(F)

    fix = np.where(V1[:, 1] < 1e-3)[0]

    L0 = compute_edges_length(V1, E)

    t0 = time.time()

    energy = []

    obj_prev = 0

    it_time = 0

    while (True):

        # energy
        obj = compute_optimization_objective(V1, F, E, x_csl, L0, w)

        if abs(obj_prev - obj) < obj_tol:
            break

        if it_time > iter:
            break

        obj_prev = obj

        energy.append(obj)

        grad = compute_optimization_objective_gradient(V1, F, E, x_csl, L0, w)

        grad[fix] = 0

        ### start of your code.

        ### end of your code.

        it_time = it_time + 1

    running_time = time.time() - t0

    return [V1, F, energy, running_time]