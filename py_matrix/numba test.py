# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from numba import jit
import numba

import sys                  # sys to add py_matrix to the path

# adding py_matrix parent folder to python path
sys.path.append('/home/pellegr/electrodynamics/st-matrix')
import py_matrix as pm

# <codecell>

def m_comp(k0, kx, ky, m_eps):

    '''Calculates the kz from the characteristic equation.

    Parameters
    ----------
    'ko'= vacuum wavevector
    'kx,ky'= in plane wavevector components
    'm_eps'= 3x3 complex dielectric tensor

    Returns
    -------
    'v_kz'= kz wavevector components
    '''

    # coefficients for the quartic equation
    A = (kx/k0)*(((m_eps[0, 2]+m_eps[2, 0])/m_eps[2, 2]) +
                 (ky/k0)*((m_eps[1, 2]+m_eps[2, 1])/m_eps[2, 2]))

    B = (((kx/k0)**2)*(1.0+m_eps[0,0]/m_eps[2,2]) +
         ((ky/k0)**2)*(1.0+m_eps[1,1]/m_eps[2,2]) +
         ((kx*ky)/((k0)**2))*(m_eps[0,1]+m_eps[1,0])/m_eps[2,2] +
         ((m_eps[0,2]*m_eps[2,0]+m_eps[1,2]*m_eps[2,1]) /
         m_eps[2,2]-m_eps[0,0]-m_eps[1,1]))

    C = (((kx**2+ky**2)/(k0**2)) *
         ((kx/k0)*(m_eps[0,2]+m_eps[2,0])/m_eps[2,2] +
         (ky/k0)*(m_eps[1,2]+m_eps[2,1])/m_eps[2,2]) +
         (kx/k0)*((m_eps[0,1]*m_eps[1,2] +
                   m_eps[1,0]*m_eps[2,1])/m_eps[2,2] -
                  (m_eps[1,1]/m_eps[2,2])*(m_eps[0,2]+m_eps[2,0])) +
         (ky/k0)*((m_eps[0,1]*m_eps[2,0] +
                  m_eps[1,0]*m_eps[0,2])/m_eps[2,2] -
                  (m_eps[0,0]/m_eps[2,2])*(m_eps[1,2]+m_eps[2,1])))

    D1 = (((kx**2+ky**2)/(k0**2))*(((kx/k0)**2)*m_eps[0,0]/m_eps[2,2] +
          ((ky/k0)**2)*m_eps[1,1]/m_eps[2,2] +
          ((kx*ky)/(k0**2)) *
          (m_eps[0,1]+m_eps[1,0])/m_eps[2,2] -
          m_eps[0,0]*m_eps[1,1]/m_eps[2,2]))

    D2 = ((kx/k0)**2)*((m_eps[0,1]*m_eps[1,0]+m_eps[0,2]*m_eps[2,0]) /
                       m_eps[2,2]-m_eps[0,0])
    D3 = ((ky/k0)**2)*((m_eps[0,1]*m_eps[1,0]+m_eps[1,2]*m_eps[2,1]) /
                       m_eps[2,2]-m_eps[1,1])
    D4 = ((kx*ky)/(k0**2))*((m_eps[0,2]*m_eps[2,1]+m_eps[2,0]*m_eps[1,2]) /
                            m_eps[2,2] -
                            m_eps[0,1]-m_eps[1,0])
    D5 = (m_eps[0,0]*m_eps[1,1]+(m_eps[0,1]*m_eps[1,2]*m_eps[2,0] +
          m_eps[1,0]*m_eps[2,1]*m_eps[0,2])/m_eps[2,2] -
          m_eps[0,1]*m_eps[1,0] -
          (m_eps[0,0]/m_eps[2,2])*m_eps[1,2]*m_eps[2,1] -
          (m_eps[1,1]/m_eps[2,2])*m_eps[0,2]*m_eps[2,0])
    D = D1+D2+D3+D4+D5

    return A,B,C,D

# <codecell>

@jit(nopython=True)
def m_comp_numba(k0, kx, ky, m_eps):

    '''Calculates the kz from the characteristic equation.

    Parameters
    ----------
    'ko'= vacuum wavevector
    'kx,ky'= in plane wavevector components
    'm_eps'= 3x3 complex dielectric tensor

    Returns
    -------
    'v_kz'= kz wavevector components
    '''

    # coefficients for the quartic equation
    A = (kx/k0)*(((m_eps[0, 2]+m_eps[2, 0])/m_eps[2, 2]) +
                 (ky/k0)*((m_eps[1, 2]+m_eps[2, 1])/m_eps[2, 2]))

    B = (((kx/k0)**2)*(1.0+m_eps[0,0]/m_eps[2,2]) +
         ((ky/k0)**2)*(1.0+m_eps[1,1]/m_eps[2,2]) +
         ((kx*ky)/((k0)**2))*(m_eps[0,1]+m_eps[1,0])/m_eps[2,2] +
         ((m_eps[0,2]*m_eps[2,0]+m_eps[1,2]*m_eps[2,1]) /
         m_eps[2,2]-m_eps[0,0]-m_eps[1,1]))

    C = (((kx**2+ky**2)/(k0**2)) *
         ((kx/k0)*(m_eps[0,2]+m_eps[2,0])/m_eps[2,2] +
         (ky/k0)*(m_eps[1,2]+m_eps[2,1])/m_eps[2,2]) +
         (kx/k0)*((m_eps[0,1]*m_eps[1,2] +
                   m_eps[1,0]*m_eps[2,1])/m_eps[2,2] -
                  (m_eps[1,1]/m_eps[2,2])*(m_eps[0,2]+m_eps[2,0])) +
         (ky/k0)*((m_eps[0,1]*m_eps[2,0] +
                  m_eps[1,0]*m_eps[0,2])/m_eps[2,2] -
                  (m_eps[0,0]/m_eps[2,2])*(m_eps[1,2]+m_eps[2,1])))

    D1 = (((kx**2+ky**2)/(k0**2))*(((kx/k0)**2)*m_eps[0,0]/m_eps[2,2] +
          ((ky/k0)**2)*m_eps[1,1]/m_eps[2,2] +
          ((kx*ky)/(k0**2)) *
          (m_eps[0,1]+m_eps[1,0])/m_eps[2,2] -
          m_eps[0,0]*m_eps[1,1]/m_eps[2,2]))

    D2 = ((kx/k0)**2)*((m_eps[0,1]*m_eps[1,0]+m_eps[0,2]*m_eps[2,0]) /
                       m_eps[2,2]-m_eps[0,0])
    D3 = ((ky/k0)**2)*((m_eps[0,1]*m_eps[1,0]+m_eps[1,2]*m_eps[2,1]) /
                       m_eps[2,2]-m_eps[1,1])
    D4 = ((kx*ky)/(k0**2))*((m_eps[0,2]*m_eps[2,1]+m_eps[2,0]*m_eps[1,2]) /
                            m_eps[2,2] -
                            m_eps[0,1]-m_eps[1,0])
    D5 = (m_eps[0,0]*m_eps[1,1]+(m_eps[0,1]*m_eps[1,2]*m_eps[2,0] +
          m_eps[1,0]*m_eps[2,1]*m_eps[0,2])/m_eps[2,2] -
          m_eps[0,1]*m_eps[1,0] -
          (m_eps[0,0]/m_eps[2,2])*m_eps[1,2]*m_eps[2,1] -
          (m_eps[1,1]/m_eps[2,2])*m_eps[0,2]*m_eps[2,0])
    D = D1+D2+D3+D4+D5

    return A,B,C,D

# <codecell>

# building the optical constant database, point the folder below to the "materials" py_matrix folder
eps_db_out=pm.mat.generate_eps_db('/home/pellegr/electrodynamics/st-matrix/py_matrix/materials/',ext='*.edb')
eps_files,eps_names,eps_db=eps_db_out['eps_files'],eps_db_out['eps_names'],eps_db_out['eps_db']

# <codecell>

wl_0=633 # incident wavelenght in nm
k0=2.0*np.pi/wl_0
theta_0=40*np.pi/1.8e2 # polar angle in radians
phi_0=0.0 # azimuthal angle radians
stack=['e_au']; # materials composing the stack, as taken from eps_db

# <codecell>

# optical constant tensor
m_eps=np.zeros((3,3),dtype=np.complex128);
e_list=pm.mat.db_to_eps(wl_0,eps_db,stack) # retrieving optical constants at wl_0 from the database
m_eps[0,0]=e_list[0] # filling dielectric tensor diagonal
m_eps[1,1]=e_list[0]
m_eps[2,2]=e_list[0]

#wavevector components
kx=k0*np.sin(theta_0)
ky=0.0

# <codecell>

%%timeit -n 1000
m_comp(k0, kx, ky, m_eps)

# <codecell>

%%timeit -n 1000
m_comp_numba(k0, kx, ky, m_eps)

# <codecell>

? %%timeit

# <codecell>


