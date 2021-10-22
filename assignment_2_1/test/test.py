import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')
from elasticsolid import *
from eigendecomposition_metric import *
eps = 1E-6

with open('test_data1.json', 'r') as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[0])
def test_rest_barycenters(data):
    v, t, rest_barycenters_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.make_rest_barycenters()
    rest_barycenters_student = es.rest_barycenters
    assert np.linalg.norm(rest_barycenters_gt - rest_barycenters_student) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[1])
def test_rest_shape_matrices(data):
    v, t, Dm_gt, Bm_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.make_rest_shape_matrices()
    Dm, Bm = es.Dm, es.Bm
    assert np.linalg.norm(Dm - Dm_gt) < eps
    assert np.linalg.norm(Bm - Bm_gt) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[2])
def test_def_barycenters(data):
    v, t, v_def, def_barycenters_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.v_def = np.array(v_def)
    es.make_def_barycenters()
    def_barycenters_student = es.def_barycenters
    assert np.linalg.norm(def_barycenters_gt - def_barycenters_student ) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[3])
def test_def_shape_matrices(data):
    v, t, v_def, Ds_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.v_def = np.array(v_def)
    es.make_def_shape_matrices()
    Ds = es.Ds
    assert np.linalg.norm(Ds - Ds_gt) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[4])
def test_update_rest_shape(data):
    v, t, v_update, rest_barycenters_gt, W0_gt, F_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.update_rest_shape(np.array(v_update))
    assert np.linalg.norm(es.rest_barycenters - rest_barycenters_gt) < eps
    assert np.linalg.norm(es.W0 - W0_gt) < eps
    assert np.linalg.norm(es.F - F_gt) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[5])
def test_update_def_shape(data):
    v, t, v_def, def_barycenters_gt, W_gt, F_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.update_def_shape(np.array(v_def))
    assert np.linalg.norm(es.def_barycenters - def_barycenters_gt) < eps
    assert np.linalg.norm(es.W - W_gt) < eps
    assert np.linalg.norm(es.F - F_gt) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[6])
def test_jacobians(data):
    v, t, v_def, Ds_gt, Bm_gt, F_gt = data
    es = ElasticSolid(np.array(v), np.array(t))
    es.v_def, es.Ds, es.Bm = np.array(v_def), np.array(Ds_gt), np.array(Bm_gt)
    es.make_jacobians()
    assert np.linalg.norm(es.F - F_gt) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[7])
def test_pinning(data):
    v, t, pin_idx, free_idx_gt, pin_mask_gt = data
    es = ElasticSolid(np.array(v), np.array(t), pin_idx = np.array(pin_idx))
    assert np.linalg.norm(es.free_idx - free_idx_gt) < eps
    assert np.linalg.norm(es.pin_mask.astype(float) - np.array(pin_mask_gt).astype(float)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[8])
def test_eig(data):
    F, eigvals_gt, eigvecs_gt = data
    eigvals, eigvecs = compute_eigendecomposition_metric(np.array(F))
    assert np.linalg.norm(eigvals - eigvals_gt) < eps
    assert np.linalg.norm(abs(np.diagonal(np.einsum('ijk, ijl -> ikl', eigvecs, eigvecs_gt), axis1 = 1, axis2 = 2)) - 1) < eps
