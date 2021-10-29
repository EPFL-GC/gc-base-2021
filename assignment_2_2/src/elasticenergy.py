import numpy as np
from numpy.core.einsumfunc import einsum
from numpy.core.fromnumeric import swapaxes

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Input:
        - young   : Young's modulus [Pa]
        - poisson : Poisson ratio
        '''
        self.young   = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu   = young / (2 * (1 + poisson))

        self.psi = None
        self.E   = None
        self.P   = None

    def make_energy_density(self, jac):
        '''
        This method computes the energy density at each tetrahedron (#t,),
        and stores the result in self.psi

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - psi : energy density per tet (#t,)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError
    
    def make_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3), and stores it in self.E

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - E : strain induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_piola_kirchhoff_stress_tensor(self, jac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        
        Updated attributes:
        - P : stress tensor induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_energy_density(self, jac):
        pass

    def make_strain_tensor(self, jac):
        pass

    def make_piola_kirchhoff_stress_tensor(self, jac):
        pass

class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.logJ = None
        self.Finv = None

    def make_energy_density(self, jac):
       pass

    def make_strain_tensor(self, jac):
        pass

    def make_piola_kirchhoff_stress_tensor(self, jac):
        '''
        Additional updated attributes:
        - logJ ; log of the determinant of the jacobians (#t,)
        - Finv : inverse of the jacobians (#t, 3, 3)
        '''
        pass