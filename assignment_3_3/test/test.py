import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')
import json
from tracer_utils import intersection_event, find_edge_point,averaged_principal_curvatures, trace
from tracer_helper import Mesh

eps=1e-6

with open('test_data.json', 'r') as infile:
    homework_data = json.load(infile)

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_data[0])
def test_intersection_event(data):
    ground_truth_t1, ground_truth_u1, ground_truth_E1, a1_orig, a1_dir, b1_orig, b1_dir, epsilon = data
    a1_orig = np.array(a1_orig, dtype = float)
    a1_dir = np.array(a1_dir, dtype = float)
    b1_orig = np.array(b1_orig, dtype = float)
    b1_dir = np.array(b1_dir, dtype = float)
    student_t1, student_u1, student_E1 = intersection_event(a1_orig, a1_dir, b1_orig, b1_dir, epsilon)
    if (ground_truth_t1 == None):
        assert ground_truth_t1 == student_t1
        assert ground_truth_u1 == student_u1
        assert ground_truth_E1 == student_E1
    else:
        assert np.linalg.norm(ground_truth_t1 - student_t1) < eps
        assert np.linalg.norm(ground_truth_u1 - student_u1) < eps
        assert np.linalg.norm(ground_truth_E1 - student_E1) < eps

@pytest.mark.timeout(0.5)
def test_find_edge_point():
    ground_truth_edge_point, ground_truth_is_boundary_point, v, f, a3_orig, a3_dir = homework_data[1]
    ground_truth_edge_point = np.array(ground_truth_edge_point, dtype=float)
    v = np.array(v, dtype=float)
    f = np.array(f, dtype=int)
    a3_orig = np.array(a3_orig, dtype=float)
    a3_dir = np.array(a3_dir, dtype=float)

    m = Mesh(v, f)
    student_edge_point, student_is_boundary_point = find_edge_point(m, a3_orig, a3_dir)
    assert np.linalg.norm(ground_truth_edge_point - student_edge_point) < eps
    assert ground_truth_is_boundary_point == student_is_boundary_point 

@pytest.mark.timeout(0.5)
def test_averaged_principal_curvatures():
    ground_truth_k1,ground_truth_k2,ground_truth_v1,ground_truth_v2,ground_truth_n, a4_orig, v, f, num_neighbors, epsilon = homework_data[2]
    ground_truth_k1 = np.array(ground_truth_k1, dtype=float)
    ground_truth_k2 = np.array(ground_truth_k2, dtype=float)
    ground_truth_v1 = np.array(ground_truth_v1, dtype=float)
    ground_truth_v2 = np.array(ground_truth_v2, dtype=float)
    ground_truth_n = np.array(ground_truth_n, dtype=float)
    a4_orig = np.array(a4_orig, dtype=float)
    v = np.array(v, dtype=float)
    f = np.array(f, dtype=int)
    m = Mesh(v, f)

    student_k1,student_k2,student_v1,student_v2,student_n = averaged_principal_curvatures(a4_orig, m, num_neighbors, epsilon)

    assert np.linalg.norm(ground_truth_k1 - student_k1) < eps
    assert np.linalg.norm(ground_truth_k2 - student_k2) < eps
    assert np.linalg.norm(ground_truth_v1 - student_v1) < eps
    assert np.linalg.norm(ground_truth_v2 - student_v2) < eps
    assert np.linalg.norm(ground_truth_n - student_n) < eps

@pytest.mark.timeout(0.5)
def test_trace_1():
    ground_truth_P1, ground_truth_A1, ground_truth_PP1, vertex_idx, v, f, num_steps, epsilon, backward_trace, num_neighbors = homework_data[3]
    ground_truth_P1 = np.array(ground_truth_P1, dtype=float)
    ground_truth_A1 = np.array(ground_truth_A1, dtype=float)
    ground_truth_PP1 = np.empty(ground_truth_PP1, dtype=float)
    v = np.array(v, dtype=float)
    f = np.array(f, dtype=int)
    m = Mesh(v, f)
    student_P1, student_A1, student_PP1 = trace(vertex_idx, m, num_steps, epsilon, True, backward_trace, num_neighbors)

    assert np.linalg.norm(ground_truth_P1 - student_P1) < eps
    assert np.linalg.norm(ground_truth_A1 - student_A1) < eps
    assert np.linalg.norm(ground_truth_PP1 - student_PP1) < eps


@pytest.mark.timeout(0.5)
def test_trace_2():
    ground_truth_P1, ground_truth_A1, ground_truth_PP1, vertex_idx, v, f, num_steps, epsilon, backward_trace, num_neighbors = homework_data[4]
    ground_truth_P1 = np.array(ground_truth_P1, dtype=float)
    ground_truth_A1 = np.array(ground_truth_A1, dtype=float)
    ground_truth_PP1 = np.empty(ground_truth_PP1, dtype=float)
    v = np.array(v, dtype=float)
    f = np.array(f, dtype=int)
    m = Mesh(v, f)
    student_P1, student_A1, student_PP1 = trace(vertex_idx, m, num_steps, epsilon, False, backward_trace, num_neighbors)
    assert np.linalg.norm(ground_truth_P1 - student_P1) < eps
    assert np.linalg.norm(ground_truth_A1 - student_A1) < eps
    assert np.linalg.norm(ground_truth_PP1 - student_PP1) < eps


