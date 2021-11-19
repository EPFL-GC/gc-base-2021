import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')
from smooth_surfaces import *

eps = 1E-6

with open('test_data.json', 'r') as infile:
    homework_datas = json.load(infile)

# Derivatives of a paraboloid 

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[0])
def test_paraboloid_points(data):
    P, a, b, c, d, e, x = data
    P = np.array(P, dtype=float)
    x = np.array(x, dtype=float)

    x_student = compute_paraboloid_points(P, a, b, c, d, e)
    assert np.linalg.norm(x_student - x) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[1])
def test_paraboloid_first_derivatives(data):
    P, a, b, c, d, e, x_u, x_v = data
    P = np.array(P, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)

    x_u_student, x_v_student = compute_paraboloid_first_derivatives(P, a, b, c, d, e)

    assert np.linalg.norm(x_u_student - x_u) < eps
    assert np.linalg.norm(x_v_student - x_v) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[2])
def test_paraboloid_second_derivatives(data):
    P, a, b, c, d, e, x_uu, x_uv, x_vv = data
    P = np.array(P, dtype=float)
    x_uu = np.array(x_uu, dtype=float)
    x_uv = np.array(x_uv, dtype=float)
    x_vv = np.array(x_vv, dtype=float)

    student_x_uu, student_x_uv, student_x_vv = compute_paraboloid_second_derivatives(P, a, b, c, d, e)

    assert np.linalg.norm(student_x_uu - x_uu) < eps
    assert np.linalg.norm(student_x_uv - x_uv) < eps
    assert np.linalg.norm(student_x_vv - x_vv) < eps

##############################

# Derivatives of a torus
@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[3])
def test_torus_points(data):
    P, R, r, x = data
    P = np.array(P, dtype=float)
    x = np.array(x, dtype=float)

    x_student = compute_torus_points(P, R, r)
    assert np.linalg.norm(x_student - x) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[4])
def test_torus_first_derivatives(data):
    P, R, r, x_u, x_v = data
    P = np.array(P, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)

    x_u_student, x_v_student = compute_torus_first_derivatives(P, R, r)

    assert np.linalg.norm(x_u_student - x_u) < eps
    assert np.linalg.norm(x_v_student - x_v) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[5])
def test_torus_second_derivatives(data):
    P, R, r, x_uu, x_uv, x_vv = data
    P = np.array(P, dtype=float)
    x_uu = np.array(x_uu, dtype=float)
    x_uv = np.array(x_uv, dtype=float)
    x_vv = np.array(x_vv, dtype=float)

    student_x_uu, student_x_uv, student_x_vv = compute_torus_second_derivatives(P, R, r)

    assert np.linalg.norm(student_x_uu - x_uu) < eps
    assert np.linalg.norm(student_x_uv - x_uv) < eps
    assert np.linalg.norm(student_x_vv - x_vv) < eps

##############################

# Shape Operator
@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[6])
def test_paraboloid_first_fundamental_form(data):
    P, a, b, c, d, e, I, x_u, x_v = data
    P = np.array(P, dtype=float)
    I = np.array(I, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    I_student = compute_first_fundamental_form(x_u, x_v)

    assert np.linalg.norm(I_student - I) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[7])
def test_torus_first_fundamental_form(data):
    P, R, r, I, x_u, x_v = data
    P = np.array(P, dtype=float)
    I = np.array(I, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    I_student = compute_first_fundamental_form(x_u, x_v)

    assert np.linalg.norm(I_student - I) < eps



@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[8])
def test_paraboloid_surface_normal(data):
    P, a, b, c, d, e, n, x_u, x_v = data
    P = np.array(P, dtype=float)
    n = np.array(n, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    n_student = compute_surface_normal(x_u, x_v)

    assert np.linalg.norm(n_student - n) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[9])
def test_torus_surface_normal(data):
    P, R, r, n, x_u, x_v = data
    P = np.array(P, dtype=float)
    n = np.array(n, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    n_student = compute_surface_normal( x_u, x_v)

    assert np.linalg.norm(n_student - n) < eps


@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[10])
def test_paraboloid_second_fundamental_form(data):
    P, a, b, c, d, e, II, x_uu, x_uv, x_vv, n = data
    P = np.array(P, dtype=float)
    II = np.array(II, dtype=float)
    x_uu = np.array(x_uu, dtype=float)
    x_uv = np.array(x_uv, dtype=float)
    x_vv = np.array(x_vv, dtype=float)
    n = np.array(n, dtype=float)

    II_student = compute_second_fundamental_form(x_uu, x_uv, x_vv, n )
    assert np.linalg.norm(II_student - II) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[11])
def test_torus_second_fundamental_form(data):
    P, R, r, II, x_uu, x_uv, x_vv, n = data
    P = np.array(P, dtype=float)
    II = np.array(II, dtype=float)
    x_uu = np.array(x_uu, dtype=float)
    x_uv = np.array(x_uv, dtype=float)
    x_vv = np.array(x_vv, dtype=float)
    n = np.array(n, dtype=float)

    II_student = compute_second_fundamental_form(x_uu, x_uv, x_vv, n)
    assert np.linalg.norm(II_student - II) < eps
    
@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[12])
def test_paraboloid_shape_operator(data):
    P, a, b, c, d, e, S, I, II = data
    P = np.array(P, dtype=float)
    S = np.array(S, dtype=float)
    I = np.array(I, dtype=float)
    II = np.array(II, dtype=float)

    S_student = compute_shape_operator(I, II)
    assert np.linalg.norm(S_student - S) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[13])
def test_torus_shape_operator(data):
    P, R, r, S, I, II = data
    P = np.array(P, dtype=float)
    S = np.array(S, dtype=float)
    I = np.array(I, dtype=float)
    II = np.array(II, dtype=float)
    
    S_student = compute_shape_operator(I, II)
    assert np.linalg.norm(S_student - S) < eps

##############################

# Principal Curvatures

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[14])
def test_paraboloid_principal_curvatures(data):
    P, a, b, c, d, e, k1, k2, e1, e2, S, x_u, x_v = data
    P = np.array(P, dtype=float)
    k1 = np.array(k1, dtype=float)
    k2 = np.array(k2, dtype=float)
    e1 = np.array(e1, dtype=float)
    e2 = np.array(e2, dtype=float)
    S = np.array(S, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    k_1_student, k_2_student, e_1_student, e_2_student = compute_principal_curvatures(S, x_u, x_v)

    assert np.linalg.norm(k_1_student - k1) < eps
    assert np.linalg.norm(k_2_student - k2) < eps
    assert np.linalg.norm(e_1_student - e1) < eps
    assert np.linalg.norm(e_2_student - e2) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[15])
def test_torus_principal_curvatures(data):
    P, R, r, k1, k2, e1, e2, S, x_u, x_v = data
    P = np.array(P, dtype=float)
    k1 = np.array(k1, dtype=float)
    k2 = np.array(k2, dtype=float)
    e1 = np.array(e1, dtype=float)
    e2 = np.array(e2, dtype=float)
    S = np.array(S, dtype=float)
    x_u = np.array(x_u, dtype=float)
    x_v = np.array(x_v, dtype=float)
    k_1_student, k_2_student, e_1_student, e_2_student = compute_principal_curvatures(S, x_u, x_v)

    assert np.linalg.norm(k_1_student - k1) < eps
    assert np.linalg.norm(k_2_student - k2) < eps
    assert np.linalg.norm(e_1_student - e1) < eps
    assert np.linalg.norm(e_2_student - e2) < eps

##############################

# Asymptotic Directions

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[16])
def test_paraboloid_asymptotic_direction(data):
    P, a, b, c, d, e, a1, a2, k1, k2, e1, e2 = data
    P = np.array(P, dtype=float)
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)
    k1 = np.array(k1, dtype=float)
    k2 = np.array(k2, dtype=float)
    e1 = np.array(e1, dtype=float)
    e2 = np.array(e2, dtype=float)
    a_1_student, a_2_student = compute_asymptotic_directions(k1, k2, e1, e2)

    assert (np.linalg.norm(a_1_student - a1) < eps and np.linalg.norm(a_2_student - a2) < eps) or (np.linalg.norm(a_1_student - a2) < eps and np.linalg.norm(a_2_student - a1) < eps)

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[17])
def test_torus_asymptotic_direction(data):
    P, R, r, a1, a2, k1, k2, e1, e2 = data
    P = np.array(P, dtype=float)
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)
    k1 = np.array(k1, dtype=float)
    k2 = np.array(k2, dtype=float)
    e1 = np.array(e1, dtype=float)
    e2 = np.array(e2, dtype=float)

    a_1_student, a_2_student = compute_asymptotic_directions(k1, k2, e1, e2)

    assert (np.linalg.norm(a_1_student - a1) < eps and np.linalg.norm(a_2_student - a2) < eps) or (np.linalg.norm(a_1_student - a2) < eps and np.linalg.norm(a_2_student - a1) < eps)


##############################
