import numpy as np
import torch
import sys as _sys
_sys.path.append("../src/")
from elasticenergy import *
from elasticsolid import *
from adjoint_sensitivity import *
from vis_utils import *
from objectives import *
from harmonic_interpolator import *
from shape_optimizer import *
from utils import *
import json, pytest

torch.set_default_dtype(torch.float64)
eps = 1E-6

with open('test_data4_beam.json', 'r') as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(2)
@pytest.mark.parametrize("data", homework_datas[0])
def test_adjoint(data):
    young, poisson, v, t, rho, pin_idx, force_mass, v_rest, v_def, dJ_dx_U, adjoint_gt  = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(torch.tensor(v), torch.tensor(np.array(t), dtype=torch.int64), ee, rho=rho, pin_idx=torch.tensor(pin_idx, dtype=torch.int64), f_mass=torch.tensor(force_mass))
    es.update_rest_shape(torch.tensor(v_rest))
    es.update_def_shape(torch.tensor(v_def))
    assert torch.linalg.norm(compute_adjoint(es, torch.tensor(dJ_dx_U)) - torch.tensor(adjoint_gt)) < eps
    
@pytest.mark.timeout(2)
@pytest.mark.parametrize("data", homework_datas[1])
def test_obj(data):
    v, vt, bv, obj_gt  = data
    assert torch.linalg.norm(objective_target_BV(torch.tensor(v), torch.tensor(vt), torch.tensor(bv)) - torch.tensor(obj_gt)) < eps

@pytest.mark.timeout(2)
@pytest.mark.parametrize("data", homework_datas[2])
def test_obj_grad(data):
    v, vt, bv, obj_grad_gt  = data
    assert torch.linalg.norm(grad_objective_target_BV(torch.tensor(v), torch.tensor(vt), torch.tensor(bv)) - torch.tensor(obj_grad_gt)) < eps

@pytest.mark.timeout(2)
@pytest.mark.parametrize("data", homework_datas[3])
def test_pt_load(data):
    young, poisson, v, t, rho, pin_idx, force_mass, v_rest, v_def, f_point_idx, f_point, f_point_gt, f_ext_gt  = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(torch.tensor(v), torch.tensor(np.array(t), dtype=torch.int64), ee, rho=rho, pin_idx=torch.tensor(pin_idx, dtype=torch.int64), f_mass=torch.tensor(force_mass))
    es.update_rest_shape(torch.tensor(v_rest))
    es.update_def_shape(torch.tensor(v_def))
    es.add_point_load(torch.tensor(f_point_idx), torch.tensor(f_point))
    assert torch.linalg.norm(es.f_point - torch.tensor(f_point_gt)) < eps
    assert torch.linalg.norm(es.f_ext - torch.tensor(f_ext_gt)) < eps
    
    
@pytest.mark.parametrize("data", homework_datas[4])
def test_set_params(data):
    young, poisson, v, t, vt_surf, rho, pin_idx, force_mass, v_rest, v_def, new_params, v_rest_gt, v_def_gt  = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(torch.tensor(v_rest), torch.tensor(np.array(t), dtype=torch.int64), ee, rho=rho, pin_idx=torch.tensor(pin_idx, dtype=torch.int64), f_mass=torch.tensor(force_mass))
    
    es.update_def_shape(torch.tensor(v_def))
    optimizer = ShapeOptimizer(es, torch.tensor(vt_surf), weight_reg=0.)
    optimizer.set_params(torch.tensor(new_params))
    assert torch.linalg.norm(optimizer.solid.v_rest - torch.tensor(v_rest_gt)) < eps
    assert torch.linalg.norm(optimizer.solid.v_def - torch.tensor(v_def_gt)) < eps

@pytest.mark.parametrize("data", homework_datas[5])
def test_line_search(data):
    young, poisson, v, t, vt_surf, rho, pin_idx, force_mass, v_rest, v_def, obj_gt, l_iter_gt, success_gt  = data

    ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(torch.tensor(v_rest), torch.tensor(np.array(t), dtype=torch.int64), ee, rho=rho, pin_idx=torch.tensor(pin_idx, dtype=torch.int64), f_mass=torch.tensor(force_mass))
    # es.update_rest_shape(torch.tensor(v_rest))
    es.update_def_shape(torch.tensor(v_def))
    optimizer = ShapeOptimizer(es, torch.tensor(vt_surf), weight_reg=0.)
    obj, l_iter, success = optimizer.line_search_step(1e-2, 10)
    assert torch.linalg.norm(obj - torch.tensor(obj_gt)) < eps
    assert l_iter == l_iter_gt
    assert success == success_gt
    