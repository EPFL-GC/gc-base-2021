# import pytest
import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')
from src.energies import *
from src.linesearch import *
from src.bfgs import *
eps = 1E-6

with open('test_data3.json', 'r') as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[0])
def test_armijo_rule(data):
    Fx = data[0]
    Fy = data[1]
    p =  np.array(data[2], dtype=float)
    grad =  np.array(data[3], dtype=float)
    c = data[4]
    alpha = data[5]
    armijo_rule_ground_truth = data[6]
    armijo_rule_student = int(evaluate_armijo_rule(Fx, Fy, p, grad, c, alpha))
    assert armijo_rule_ground_truth == armijo_rule_student

def func(x):
    return np.linalg.norm(x) ** 2

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[1])
def test_backtracking_line_search(data):
    p =  np.array(data[0], dtype=float)
    grad = np.array(data[1], dtype=float)
    x = np.array(data[2], dtype=float)
    theta = data[3]
    beta = data[4]
    c = data[5]
    backtracking_line_search_ground_truth = data[6]
    backtracking_line_search_student = backtracking_line_search(p, grad, x, theta, beta, c, func)

    assert np.linalg.norm(backtracking_line_search_ground_truth - backtracking_line_search_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[2])
def test_compute_approximate_hessian_matrix(data):
    sk = np.array(data[0], dtype=float)
    yk = np.array(data[1], dtype=float)
    Bk = np.array(data[2], dtype=float)

    newBk_student = compute_approximate_hessian_matrix(sk, yk, Bk)
    newBk_ground_truth = np.array(data[3], dtype=float)
    assert np.linalg.norm(newBk_student - newBk_ground_truth) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[3])
def test_compute_inverse_approximate_hessian_matrix(data):
    sk = np.array(data[0], dtype=float)
    yk = np.array(data[1], dtype=float)
    invBk = np.array(data[2], dtype=float)
    inv_newBk_student = compute_inverse_approximate_hessian_matrix(sk, yk, invBk)
    inv_newBk_ground_truth = np.array(data[3], dtype=float)
    assert np.linalg.norm(inv_newBk_student - inv_newBk_ground_truth) < eps
