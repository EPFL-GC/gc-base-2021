import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import *
from shear import *

eps = 1E-6

with open(os.path.join(os.path.dirname(__file__), "test_data1.json"), "r") as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[0])
def test_faces_centroid(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    centroid_ground_truth = np.array(data[2])
    centroid_student = compute_faces_centroid(V, F)
    assert np.linalg.norm(centroid_ground_truth - centroid_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[1])
def test_faces_area(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    area_ground_truth = np.array(data[2])
    area_student = compute_faces_area(V, F)
    assert np.linalg.norm(area_ground_truth - area_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[2])
def test_mesh_centroid(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    centroid_ground_truth = np.array(data[2])
    centroid_student = compute_mesh_centroid(V, F)
    assert np.linalg.norm(centroid_ground_truth - centroid_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[3])
def test_mesh_area(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    area_ground_truth = np.array(data[2])
    area_student = compute_mesh_area(V, F)
    assert abs(area_ground_truth - area_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[4])
def test_center_support_line(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    support_line_ground_truth = np.array(data[2])
    support_line_student = compute_center_support_line(V)
    assert abs(support_line_ground_truth - support_line_student) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[5])
def test_shear_tranformation(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    nu = data[2]
    V_shear_ground_truth = np.array(data[3], dtype=float)
    V_shear_student = shear_transformation(V, nu)
    assert np.linalg.norm(V_shear_student - V_shear_ground_truth) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[6])
def test_shear_equilibrium(data):
    V = np.array(data[0], dtype=float)
    F = np.array(data[1], dtype=int)
    x_csl = data[2]
    V_shear_ground_truth = np.array(data[3], dtype=float)
    V_shear_student = shear_equilibrium(V, F, x_csl)
    assert np.linalg.norm(V_shear_student - V_shear_ground_truth) < eps
