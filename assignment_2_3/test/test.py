import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')
from elasticsolid import *
from elasticenergy import *
eps = 1E-6

with open('test_data3.json', 'r') as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[0])
def test_linear_dE(data):
    young, poisson, F, dF, dE_gt = data
    ee = LinearElasticEnergy(young, poisson)
    ee.make_differential_strain_tensor(np.array(F), np.array(dF))
    assert np.linalg.norm(ee.dE - np.array(dE_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[1])
def test_linear_dP(data):
    young, poisson, F, dF, dE, dP_gt = data
    ee = LinearElasticEnergy(young, poisson)
    ee.dE = np.array(dE)
    ee.make_differential_piola_kirchhoff_stress_tensor(np.array(F), np.array(dF))
    assert np.linalg.norm(ee.dP - np.array(dP_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[2])
def test_neo_dP(data):
    young, poisson, F, dF, dP_gt = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    ee.logJ = np.log(np.linalg.det(np.array(F)))
    ee.Finv = np.linalg.inv(np.array(F))
    ee.make_differential_piola_kirchhoff_stress_tensor(np.array(F), np.array(dF))
    assert np.linalg.norm(ee.dP - np.array(dP_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[3])
def test_force_differentials(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, dx, df_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=pin_idx, f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    
    assert np.linalg.norm(es.compute_force_differentials(np.array(dx)) - np.array(df_gt)) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[4])
def test_CG(data):
    A, RHS, dx_gt = data
    A = np.array(A)
    def LHS(dx_):
        return A@dx_
    assert np.linalg.norm(conjugate_gradient(LHS, np.array(RHS)) - np.array(dx_gt)) < eps

@pytest.mark.timeout(1)
@pytest.mark.parametrize("data", homework_datas[5])
def test_lhs_rhs(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, dx, lhsdx_gt, rhs_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=pin_idx, f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    es.equilibrium_step(max_l_iter=0)

    
    assert np.linalg.norm(es.LHS(np.array(dx)) - np.array(lhsdx_gt)) < eps
    assert np.linalg.norm(es.RHS - rhs_gt) < eps
