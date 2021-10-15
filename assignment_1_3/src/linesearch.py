from energies import *
from optimization import compute_optimization_objective, compute_optimization_objective_gradient
import numpy as np
import time
import igl

def evaluate_armijo_rule(f_x, f_x1, p, grad, c, alpha):
    """
       Check the armijo rule, return true if the armijo condition is satisfied

        Input:
        - f_x : float
            The function value at x
        - f_x1 : float
            The function value at x_(k+1) = x_k + alpha * p_k
        - p: np.array(2 * #V, 1)
            The flatten search direction
            Here, we use the flatten the search direction.
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.
        - grad: np.array(2 * #V, 1)
            The gradient of the function at x
            Here, we use the flatten the gradient
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.
        - c: float
            The coefficient for armijo condition
        - alpha: float
            The current step size

        Output:
        - condition: bool
            True if the armijio condition is satisfied
    """

    return True

def backtracking_line_search(p, grad, x, theta, beta, c, f, *arg):
    """
        Computes the step size for p that satisfies the armijio condition.

        Input:
        - p: np.array(2 * #V, 1)
            The flatten search direction
            Here, we use the flatten the search direction.
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.
        - grad: np.array(2 * #V, 1)
            The gradient of the function at x
            Here, we use the flatten the gradient
            The flatten process asks the all x coordinates to be first
            then all y cooridnates to be the second.
        - x : np.array(#V * 2, 1)
            The array of optimization variables
            x = V[:, 0 : 2].flatten()
        - theta: float
            The initial step size
        - beta : float
            The backtracking ratio, alpha = beta * alpha
        - c: float
            The coefficient for armijo condition
        - f: function
            The objective function (i.e., optimization.compute_optimization_objective)
        - *arg: parameters
            The rest parameters for the function f except the its first variables Vx

        Output:
        - alpha: float
           The step size for p that satisfies the armijio condition
    """

    alpha = theta

    return alpha

def gradient_descent_with_line_search(V, F, x_csl, w, obj_tol, theta, beta, c, iter):
    """
    Find equilibrium shape by using gradient descent with backtracking line search

    Input:
    - V : np.array (#V, 3)
        The array of vertices positions.
        Contains the coordinates of the i-th vertex in i-th row.
    - F : np.array (#F, 3)
        The array of triangle faces.
    - x_csl : float
        The x coordinate of the center of the support line.
    - w : float
        The weight for shape preservation energy.
    - obj_tol: float
        The termination condition for optimization.
        The program stop running if
        the absolute difference between the objectives of two consecutive iterations is smaller than obj_tol
    - theta : float
        The initial gradient descent step size.
    - beta : float
        The backtracking ratio, alpha = beta * alpha
    - c: float
        The coefficient for armijo condition.
     - iter : int
        The maximum number of iteration for gradient descent.

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

    while(True):

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
