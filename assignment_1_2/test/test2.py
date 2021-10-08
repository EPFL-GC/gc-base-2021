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
eps = 1E-6

with open('test_data2.json', 'r') as infile:
    homework_datas = json.load(infile)

# @pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[0])
def test_shape_energy_correctness(data):
    V = np.array(data[0], dtype=float)
    E = np.array(data[1], dtype=int)
    l0 = np.array(data[2], dtype=float)
    shape_energy_ground_truth = np.array(data[3])
    shape_energy_student = compute_shape_energy(V, E, l0)
    assert np.linalg.norm(shape_energy_ground_truth - shape_energy_student) < eps

# @pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[1])
def test_equilibrium_engery_correctness(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    x_csl = data[2]
    equilibrium_engery_ground_truth = np.array(data[3])
    equilibrium_engery_student = compute_equilibrium_energy(V, F, x_csl)
    assert np.linalg.norm(equilibrium_engery_ground_truth - equilibrium_engery_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[2])
def test_faces_area_gradient_correctness(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    faces_area_gradient_ground_truth_x = np.array(data[2])
    faces_area_gradient_ground_truth_y = np.array(data[3])
    [faces_area_gradient_student_x,  faces_area_gradient_student_y] = compute_faces_area_gradient(V, F)
    assert np.linalg.norm(faces_area_gradient_ground_truth_x - faces_area_gradient_student_x) < eps \
           and np.linalg.norm(faces_area_gradient_ground_truth_y - faces_area_gradient_student_y) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[3])
def test_equilibrium_energy_gradient_correctness(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    x_csl = data[2]
    equilibrium_energy_gradient = np.array(data[3])
    equilibrium_energy_student = compute_equilibrium_energy_gradient(V, F, x_csl)
    assert np.linalg.norm(equilibrium_energy_gradient - equilibrium_energy_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[4])
def test_shape_energy_gradient_correctness(data):
    V = np.array(data[0], dtype=float)
    E = np.array(data[1], dtype=int)
    l0 = np.array(data[2], dtype=float)
    shape_energy_gradient_ground_truth = np.array(data[3])
    shape_energy_gradient_student = compute_shape_energy_gradient(V, E, l0)
    assert np.linalg.norm(shape_energy_gradient_ground_truth - shape_energy_gradient_student) < eps

n = 100000
F_big = np.arange(3 * n).reshape((n, 3))
V_big = np.random.random((3 * n, 3))
l0_big = np.zeros(3 * n)
E_big = igl.edges(F_big)

@pytest.mark.timeout(1)
def test_shape_energy_timing():
    compute_shape_energy(V_big, E_big, l0_big)

@pytest.mark.timeout(1)
def test_faces_area_gradient_timing():
    compute_faces_area_gradient(V_big, F_big)

@pytest.mark.timeout(1)
def test_equilibrium_energy_gradient_timing():
    compute_equilibrium_energy_gradient(V_big, F_big, 0)

@pytest.mark.timeout(1)
def test_shape_energy_gradient_timing():
    compute_shape_energy_gradient(V_big, E_big, l0_big)