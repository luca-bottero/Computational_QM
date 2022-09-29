import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.linalg import *

class TISE_1D():
    def __init__(self, x, potential, n_eigs=1, units='AU'):
        self.x = x
        self.x_min = x[0]
        self.x_max = x[-1]
        self.Nx = x.shape[0]
        self.units = units
        self._init_units()

        if potential is not None:
            self.pot = potential(x)

        self.n_eigs = n_eigs
        self.dx = self.x[1] - self.x[0]

        self.lapl = sparse.eye(self.Nx, self.Nx, format='lil') * (-2)
        for i in range(self.Nx -1):
            self.lapl[i, i+1] = 1
            self.lapl[i+1, i] = 1
        self.lapl /= self.dx**2

        self.create_hamiltonian()

    def _init_units(self):
        if self.units == 'SI':
            self.hbar   = const.hbar
            self.m_e    = const.m_e
            #self.eps_0  = const.epsilon_0

        elif self.units == 'AU':
            # atomic units
            self.hbar   = 1.
            self.m_e    = 1.
            #self.eps_0  = 1.


    def create_hamiltonian(self):
        self.hamiltonian = -(self.hbar**2/self.m_e)*self.lapl + sparse.diags(self.pot)
        #self.hamiltonian = sparse.csc_matrix(self.hamiltonian)


    def solve(self, real_eigs = True, method='tridiagonal', **kwargs):
        if method == 'tridiagonal':
            diag = self.hamiltonian.diagonal(0)
            offdiag = self.hamiltonian.diagonal(1)
            self.eigenvalues, self.eigenfunctions = la.eigh_tridiagonal(diag, offdiag, **kwargs)
        elif method == 'eigsh':
            self.eigenvalues, self.eigenfunctions = sla.eigsh(self.hamiltonian, k=self.n_eigs, which='SM')

        self._postprocess(real_eigs)

    def _postprocess(self, real_eigs):
        self.eigenvalues /= 2
        if real_eigs:
            self.eigenvalues = np.real(self.eigenvalues)
        
        self._normalize_eigenfunctions()
        
        idxs = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idxs]
        self.eigenfunctions = self.eigenfunctions[:,idxs]

    def _normalize_eigenfunctions(self):
        for i in range(self.eigenvalues.shape[0]):
            self.eigenfunctions[:,i] /= np.sqrt(np.trapz(np.conj(self.eigenfunctions[:,i])*self.eigenfunctions[:,i], self.x))

    def plot_eigenvalues(self):
        pass

class TISE_1D_RADIAL(TISE_1D):
    def __init__(self, x, potential, n_eigs=1, units='SI', l=0):

        self.l = l
        super().__init__(x, potential, n_eigs, units)

        if not (self.x > 0.).all():
           raise ValueError('Coordinate points must be strictly positive')
        
    def create_hamiltonian(self):
        self.hamiltonian = ( -(self.hbar**2/self.m_e)*self.lapl + sparse.diags(self.pot) 
                            + sparse.diags((self.hbar**2/self.m_e)*(self.l + 1)*self.l/self.x**2) )
        #self.hamiltonian = sparse.csc_matrix(self.hamiltonian)

    def _normalize_eigenfunctions(self):
        for i in range(self.eigenvalues.shape[0]):
            self.eigenfunctions[:,i] /= np.sqrt(4*const.pi*np.abs(np.trapz(
                        self.x**2*np.conj(self.eigenfunctions[:,i])*self.eigenfunctions[:,i], self.x)))