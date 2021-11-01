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

with open('test_data2.json', 'r') as infile:
    homework_datas = json.load(infile)

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[0])
def test_linear_psi(data):
    young, poisson, F, psi_gt = data
    ee = LinearElasticEnergy(young, poisson)
    print(np.array(F).shape)
    ee.make_strain_tensor(np.array(F))
    ee.make_energy_density(np.array(F))
    assert np.linalg.norm(ee.psi - np.array(psi_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[1])
def test_linear_strain(data):
    young, poisson, F, E_gt = data
    ee = LinearElasticEnergy(young, poisson)
    ee.make_strain_tensor(np.array(F))
    print(np.array(E_gt).shape, np.array(E_gt)[0])
    assert np.linalg.norm(ee.E - np.array(E_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[2])
def test_linear_stress(data):
    young, poisson, F, P_gt = data
    ee = LinearElasticEnergy(young, poisson)
    ee.make_strain_tensor(np.array(F))
    ee.make_piola_kirchhoff_stress_tensor(np.array(F))
    assert np.linalg.norm(ee.P - np.array(P_gt)) < eps


@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[3])
def test_neo_psi(data):
    young, poisson, F, psi_gt = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    ee.make_energy_density(np.array(F))
    assert np.linalg.norm(ee.psi - np.array(psi_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[4])
def test_neo_stress(data):
    young, poisson, F, P_gt, logJ_gt, Finv_gt = data
    ee = NeoHookeanElasticEnergy(young, poisson)
    ee.make_piola_kirchhoff_stress_tensor(np.array(F))
    assert np.linalg.norm(ee.P - np.array(P_gt)) < eps
    assert np.linalg.norm(ee.logJ - np.array(logJ_gt)) < eps
    assert np.linalg.norm(ee.Finv - np.array(Finv_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[5])
def test_energy_el(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, ee_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=np.array(pin_idx), f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    assert np.linalg.norm(es.energy_el - np.array(ee_gt)) < eps
    
@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[6])
def test_forces_el(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, ef_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=np.array(pin_idx), f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    assert np.linalg.norm(es.f - np.array(ef_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[7])
def test_energy_ext(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, exte_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=np.array(pin_idx), f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    assert np.linalg.norm(es.energy_ext - np.array(exte_gt)) < eps

@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("data", homework_datas[8])
def test_forces_ext(data):
    etype, young, poisson, v, t, rho, pin_idx, force_mass, v_def, fext_gt, fvol_gt  = data
    if etype == "linear":
        ee = LinearElasticEnergy(young, poisson)
    else:
        ee = NeoHookeanElasticEnergy(young, poisson)
    
    es = ElasticSolid(np.array(v), np.array(t), ee, rho=rho, pin_idx=np.array(pin_idx), f_mass=np.array(force_mass))
    es.update_def_shape(np.array(v_def))
    assert np.linalg.norm(es.f_ext - np.array(fext_gt)) < eps
    assert np.linalg.norm(es.f_vol - np.array(fvol_gt)) < eps
