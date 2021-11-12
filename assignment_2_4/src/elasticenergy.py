import numpy as np
import torch

torch.set_default_dtype(torch.float64)

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Input:
        - young   : Young's modulus [Pa]
        - poisson : Poisson ratio
        '''
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.mu = young / (2 * (1 + poisson))

        self.E   = None
        self.dE  = None
        self.psi = None
        self.P   = None
        self.dP  = None
    
    def make_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3), and stores it in self.E

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_strain_tensor(self, jac, dJac):
        '''
        This method computes the differential of strain tensor (#t, 3, 3), 
        and stores it in self.dE

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_energy_density(self, jac):
        '''
        This method computes the energy density at each tetrahedron (#t,),
        and stores the result in self.psi

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_piola_kirchhoff_stress_tensor(self, jac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac : jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):
        '''
        This method computes the stress tensor (#t, 3, 3), and stores it in self.P

        Input:
        - jac  : jacobian of the deformation (#t, 3, 3)
        - dJac : differential of the jacobian of the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)

    def make_energy_density(self, jac):
        
        # First, update the strain tensor
        self.make_strain_tensor(jac)

        # psi = mu*E:E + lbda/2*Tr(E)^2
        self.psi = (self.mu * torch.einsum('mij,mij->m', self.E, self.E) +
                    self.lbda/2 * torch.einsum('mii->m', self.E) ** 2)
        pass

    def make_strain_tensor(self, jac):
        eye = torch.zeros((len(jac), 3, 3))
        for i in range(3):
            eye[:, i, i] = 1
        
        # E = 1/2*(F + F^T) - I
        self.E = 0.5*(torch.swapaxes(jac, 1, 2) + jac) - eye
        pass
    
    def make_piola_kirchhoff_stress_tensor(self, jac):

        # First, update the strain tensor
        self.make_strain_tensor(jac)

        tr  = torch.einsum('ijj->i', self.E)
        eye = torch.zeros((len(self.E), 3, 3))
        for i in range(3):
            eye[:, i, i] = tr 

        # P = 2*mu*E + lbda*tr(E)*I = 
        self.P = 2 * self.mu * self.E + self.lbda * eye
        pass

    def make_differential_strain_tensor(self, jac, dJac):
        # dE = 1/2*(dF + dF^T)
        self.dE = 0.5 * (dJac + torch.swapaxes(dJac, 1, 2))
        pass

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):

        # First, update the differential of the strain tensor, 
        # and the strain tensor
        self.make_strain_tensor(jac)
        self.make_differential_strain_tensor(jac, dJac)

        # Diagonal matrix
        dtr = torch.einsum('ijj->i', self.dE)
        dI  = torch.zeros((len(jac), 3, 3))
        for i in range(3):
            dI[:, i, i] = dtr

        # dP = 2*mu*dE + lbda*tr(dE)*I
        self.dP = 2 * self.mu * self.dE + self.lbda * dI
        pass


class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.logJ = None
        self.Finv = None

    def make_energy_density(self, jac):
        
        # First, update the strain tensor
        self.make_strain_tensor(jac)

        # J   = det(F)
        # I1  = Tr(F^T.F)
        # psi = mu/2*(I1 - 3 - 2*log(J)) + lbda/2*log(J)^2
        logJ     = torch.log(torch.linalg.det(jac))
        I1       = torch.einsum('mji,mji->m', jac, jac)
        self.psi = self.mu/2 * (I1 - 3 - 2*logJ) + self.lbda/2 * logJ**2
        pass

    def make_strain_tensor(self, jac):
        pass

    def make_piola_kirchhoff_stress_tensor(self, jac):
        self.logJ = torch.log(torch.linalg.det(jac))
        # First invert, then transpose
        self.Finv = torch.linalg.inv(jac)
        FinvT = torch.swapaxes(self.Finv, 1, 2)

        # P = mu*(F - F^{-T}) + lbda*log(J)*F^{-T}
        self.P = (self.mu * (jac - FinvT) + self.lbda * torch.einsum('i,ijk->ijk', self.logJ, FinvT))
        pass

    def make_differential_strain_tensor(self, jac, dJac):
        pass

    def make_differential_piola_kirchhoff_stress_tensor(self, jac, dJac):

        # To be reused below
        logJ     = self.logJ.reshape(-1, 1, 1) # (#t, 1, 1) for shape broadcasting
        FinvT    = torch.swapaxes(self.Finv, 1, 2)
        Fprod    = torch.einsum("mij, mjk, mkl -> mil", FinvT, torch.swapaxes(dJac, 1, 2), FinvT)
        trFinvdF = torch.einsum("mij, mji -> m", self.Finv, dJac)

        # dP = mu*dF + (mu-lbda*log(J))*F^{-T}.dF^T.F^{-T} + lbda*tr(F^{-1}.dF)*F^{-T}
        self.dP = (self.mu * dJac + 
                   (self.mu - self.lbda * logJ) * Fprod + 
                   self.lbda * torch.einsum("m, mij -> mij", trFinvdF, FinvT)
                  )
        pass