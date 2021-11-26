import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')


import utils, laplacian_utils, mean_curvature_flow, remesher_helper
from utils import parse_input_mesh, normalize_area, get_diverging_colors, remesh
from mean_curvature_flow import MCF

import copy
import numpy.linalg as la

import fitting 

epsilon1 = 5e-2
epsilon2 = 1e-3

eps = 1E-6

with open('test_data.json', 'r') as infile:
    homework_data = json.load(infile)

v = np.array(homework_data[0], dtype=float)
f = np.array(homework_data[1], dtype=int)
num_bdry_vx, num_intr_vx = int(homework_data[2]), int(homework_data[3])
curr_mcf = MCF(num_bdry_vx, num_intr_vx)
curr_mcf.bbox_diagonal = igl.bounding_box_diagonal(v)


@pytest.mark.timeout(0.5)
def test_mass_matrix():
    student_mass_matrix = laplacian_utils.compute_mass_matrix(v, f).toarray()
    assert np.linalg.norm(student_mass_matrix - np.array(homework_data[4], dtype=float)) < eps

@pytest.mark.timeout(0.5)
def test_cotan():
    cotan, a, b, c, A = homework_data[5]
    student_cotan = laplacian_utils.compute_cotangent(a, b, c, A)
    assert np.linalg.norm(student_cotan - cotan) < eps

@pytest.mark.timeout(0.5)
def test_laplacian_matrix():
    student_laplacian_matrix = laplacian_utils.compute_laplacian_matrix(v, f).toarray()
    assert np.linalg.norm(student_laplacian_matrix - np.array(homework_data[6], dtype=float)) < eps

@pytest.mark.timeout(0.5)
def test_average_mean_curvature():
    data = homework_data[7]
    laplace_v = copy.deepcopy(v)
    curr_mcf.update_system(laplace_v, f)
    assert np.linalg.norm(curr_mcf.average_mean_curvature - data) < eps

@pytest.mark.timeout(0.5)
def test_laplace_solution():
    data = homework_data[8]
    laplace_v = copy.deepcopy(v)
    curr_mcf.solve_laplace_equation(laplace_v, f)
    assert np.linalg.norm(laplace_v - np.array(data, dtype=float)) < eps

@pytest.mark.timeout(0.5)
def test_meet_sc():
    data = homework_data[9]
    student_sc = [curr_mcf.meet_stopping_criteria([0.001, 0.001], epsilon1, epsilon2), curr_mcf.meet_stopping_criteria([1, 0.5], epsilon1, epsilon2)]

    assert student_sc == data

@pytest.mark.timeout(0.5)
def test_mean_curvature_flow():
    data = np.array(homework_data[10], dtype=float)
    _, student_mean_curvature_flow = curr_mcf.run_mean_curvature_flow(v, f, 100, epsilon1, epsilon2)

    assert np.linalg.norm(student_mean_curvature_flow - data) < eps

@pytest.mark.timeout(0.5)
def test_average_mean_curvature():
    data = homework_data[7]
    laplace_v = copy.deepcopy(v)
    curr_mcf.update_system(laplace_v, f)
    assert np.linalg.norm(curr_mcf.average_mean_curvature - data) < eps

@pytest.mark.timeout(0.5)
def test_laplace_solution():
    data = homework_data[8]
    laplace_v = copy.deepcopy(v)
    curr_mcf.solve_laplace_equation(laplace_v, f)
    assert np.linalg.norm(laplace_v - np.array(data, dtype=float)) < eps

@pytest.mark.timeout(0.5)
def test_meet_sc():
    data = homework_data[9]
    student_sc = [curr_mcf.meet_stopping_criteria([0.001, 0.001], epsilon1, epsilon2), curr_mcf.meet_stopping_criteria([1, 0.5], epsilon1, epsilon2)]

    assert student_sc == data

@pytest.mark.timeout(0.5)
def test_mean_curvature_flow():
    data = np.array(homework_data[10], dtype=float)
    _, student_mean_curvature_flow = curr_mcf.run_mean_curvature_flow(v, f, 100, epsilon1, epsilon2)

    assert np.linalg.norm(student_mean_curvature_flow - data) < eps


fitting_v = np.array(homework_data[11], dtype=float)
fitting_f = np.array(homework_data[12], dtype=int)

@pytest.mark.timeout(0.5)
def test_osculating_paraboloids():
    [paraboloids_parameters, e1, e2, e3] = homework_data[13]
    paraboloids_parameters = np.array(paraboloids_parameters)
    e1 = np.array(e1)
    e2 = np.array(e2)
    e3 = np.array(e3)
    student_paraboloids_parameters = fitting.compute_osculating_paraboloids(fitting_v, fitting_f, e1, e2, e3)
    assert np.linalg.norm(student_paraboloids_parameters - paraboloids_parameters) < eps

@pytest.mark.timeout(0.5)
def test_osculating_paraboloid_first_derivatives():
    [op_fd_x, op_fd_y, paraboloids_parameters] = homework_data[14]
    op_fd_x = np.array(op_fd_x)
    op_fd_y = np.array(op_fd_y)
    paraboloids_parameters = np.array(paraboloids_parameters)
    student_fd_x, student_fd_y = fitting.compute_osculating_paraboloid_first_derivatives(paraboloids_parameters)
    assert np.linalg.norm(student_fd_x - op_fd_x) < eps
    assert np.linalg.norm(student_fd_y - op_fd_y) < eps

@pytest.mark.timeout(0.5)
def test_osculating_paraboloid_second_derivatives():
    [op_sd_xx, op_sd_xy, op_sd_yy, paraboloids_parameters] = homework_data[15]
    op_sd_xx = np.array(op_sd_xx)
    op_sd_xy = np.array(op_sd_xy)
    op_sd_yy = np.array(op_sd_yy)
    paraboloids_parameters = np.array(paraboloids_parameters)
    student_sd_xx, student_sd_xy, student_sd_yy = fitting.compute_osculating_paraboloid_second_derivatives(paraboloids_parameters)

    assert np.linalg.norm(student_sd_xx - op_sd_xx) < eps
    assert np.linalg.norm(student_sd_xy - op_sd_xy) < eps
    assert np.linalg.norm(student_sd_yy - op_sd_yy) < eps

@pytest.mark.timeout(0.5)
def test_mesh_principal_curvatures():
    k1, k2, d1, d2 = homework_data[16]
    k1 = np.array(k1)
    k2 = np.array(k2)
    d1 = np.array(d1)
    d2 = np.array(d2)
    student_k1, student_k2, student_d1, student_d2 = fitting.compute_mesh_principal_curvatures(fitting_v, fitting_f)
    assert np.linalg.norm(k1 - student_k1) < eps
    assert np.linalg.norm(k2 - student_k2) < eps
    assert np.linalg.norm(d1 - student_d1) < eps
    assert np.linalg.norm(d2 - student_d2) < eps
