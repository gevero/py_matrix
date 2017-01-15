# -*- coding: utf-8 -*-

'''moe.py - mAGNETO oPTICS eLLIPSOID - contains subroutines to solve the
problem for a MO ellipsoid of a size comparable with the wavelength.
The code is inspired by the following paper:

"Maccaferri N, Gonzalez-Diaz JB, Bonetti S, et al. (2013) Polarizability and
magnetoplasmonic properties of magnetic general nanoellipsoids.
Opt Express 21:9875–9889. doi: 10.1364/OE.21.009875 '''

import numpy as np
import numpy.linalg as np_l
import scipy as sp
import scipy.integrate as sp_i


def f_V(a_x,a_y,a_z):
    '''ellipsoid volume

    Parameters
    ----------
    'a_x,a_y,a_z' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'V' = the volume of the ellipsoid'''

    V = 4.0*np.pi*a_x*a_y*a_z/3.0

    return V


def f_L(q,a_1,a_2,a_3):
    '''integrand of the static geometrical tensor

    Parameters
    ----------
    'q' = free variable for the geometrical tensor
    'a_1,a_2,a_3' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'L' = integrand of the static geometrical tensor'''

    L = sp.power(q+a_1**2,-1.5)*sp.power(q+a_2**2,-0.5)*sp.power(q+a_3**2,-0.5)

    return L


def f_Dx(t,z,a_x,a_y,a_z):
    '''integrand of the Dx component for the dynamic geometrical tensor

    Parameters
    ----------
    't,z' = theta and z cylindrical coordinates
    'a_1,a_2,a_3' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'Dx' = integrand of the Dx component for the dynamic geometrical tensor'''

    a = 2.0*(a_x**2)*(np.cos(t)**2) + (a_y**2)*(np.sin(t)**2)
    b = (a_z**2)*z**2
    c = (a_x**2)*(np.cos(t)**2) + (a_y**2)*(np.sin(t)**2)
    d = (a_z**2)*z**2

    s_1 = (b*c-2.0*a*d)/np.sqrt(d)
    s_2 = (a*(2.0*d+c*(1-z**2)) - b*c)/np.sqrt(d+c*(1-z**2))

    Dx = (s_1+s_2)/(c**2)

    return Dx


def f_Dy(t,z,a_x,a_y,a_z):
    '''integrand of the Dy component for the dynamic geometrical tensor

    Parameters
    ----------
    't,z' = theta and z cylindrical coordinates
    'a_1,a_2,a_3' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'Dy' = integrand of the Dy component for the dynamic geometrical tensor'''

    a = (a_x**2)*(np.cos(t)**2) + 2.0*(a_y**2)*(np.sin(t)**2)
    b = (a_z**2)*z**2
    c = (a_x**2)*(np.cos(t)**2) + (a_y**2)*(np.sin(t)**2)
    d = (a_z**2)*z**2

    s_1 = (b*c-2.0*a*d)/np.sqrt(d)
    s_2 = (a*(2.0*d+c*(1-z**2)) - b*c)/np.sqrt(d+c*(1-z**2))

    Dy = (s_1+s_2)/(c**2)

    return Dy


def f_Dz(t,z,a_x,a_y,a_z):
    '''integrand of the Dz component for the dynamic geometrical tensor

    Parameters
    ----------
    't,z' = theta and z cylindrical coordinates
    'a_1,a_2,a_3' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'Dz' = integrand of the Dz component for the dynamic geometrical tensor'''

    a = (a_x**2)*(np.cos(t)**2) + (a_y**2)*(np.sin(t)**2)
    b = 2.0*(a_z**2)*z**2
    c = (a_x**2)*(np.cos(t)**2) + (a_y**2)*(np.sin(t)**2)
    d = (a_z**2)*z**2

    s_1 = (b*c-2.0*a*d)/np.sqrt(d)
    s_2 = (a*(2.0*d+c*(1-z**2)) - b*c)/np.sqrt(d+c*(1-z**2))

    Dz = (s_1+s_2)/(c**2)

    return Dz


def m_L(a_x,a_y,a_z):
    '''calculates the static geometrical tensor of the ellipsoid:
       to be computed once for every geometry

    Parameters
    ----------
    'a_x,a_y,a_z' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'L' = geometrical tensor of the ellipsoid'''

    L_x_int,err = sp_i.quad(lambda q: f_L(q,a_x,a_y,a_z), 0.0, np.inf)
    L_y_int,err = sp_i.quad(lambda q: f_L(q,a_y,a_x,a_z), 0.0, np.inf)
    L_z_int,err = sp_i.quad(lambda q: f_L(q,a_z,a_y,a_x), 0.0, np.inf)
    L_y = 0.5*(a_x*a_y*a_z)*L_y_int
    L_x = 0.5*(a_x*a_y*a_z)*L_x_int
    L_z = 0.5*(a_x*a_y*a_z)*L_z_int

    L = np.array([[L_x,0.0,0.0],[0.0,L_y,0.0],[0.0,0.0,L_z]])

    return L


def m_D(a_x,a_y,a_z):
    '''calculates the dynamic geometrical tensor of the ellipsoid:
       to be computed once for every geometry

    Parameters
    ----------
    'a_x,a_y,a_z' = three axes of the ellipsoid (in nm)

    Returns
    -------
    'D' = dynamic geometrical tensor of the ellipsoid'''

    D_x_int,err = sp_i.nquad(lambda t,z: f_Dx(t,z,a_x,a_y,a_z),
                             [[0,2.0*np.pi],[0.0,1.0]])
    D_y_int,err = sp_i.nquad(lambda t,z: f_Dy(t,z,a_x,a_y,a_z),
                             [[0,2.0*np.pi],[0.0,1.0]])
    D_z_int,err = sp_i.nquad(lambda t,z: f_Dz(t,z,a_x,a_y,a_z),
                             [[0,2.0*np.pi],[0.0,1.0]])
    D_y = (0.75/np.pi)*D_y_int
    D_x = (0.75/np.pi)*D_x_int
    D_z = (0.75/np.pi)*D_z_int

    D = np.array([[D_x,0.0,0.0],[0.0,D_y,0.0],[0.0,0.0,D_z]])

    return D


def m_alpha(m_L,m_D,V,m_e1,m_e2,w_l):
    '''calculates polarizability tensor of the ellipsoid

    Parameters
    ----------
    'm_L,m_D' = static and dynamic geometrical tensor of the ellipsoid
    'V' = volume of the ellipsoind
    'm_e1,m_e2' = diagonal tensor of the embedding matrix,
                  Magneto Optical tensor of the ellipsoid
    'w_l' = incident wavelength

    Returns
    -------
    'alpha' = polarizability tensor of the ellipsoid'''

    m_I = np.identity(3)
    k = 2.0*np.pi*np.sqrt(m_e1[2,2])/w_l

    m_1 = m_L - ((k**2)*V/(4.0*np.pi))*m_D - 1j*((k**3)*V/(6.0*np.pi))*m_I
    m_2 = m_e2-m_e1
    m_3 = m_I+np.dot(np.dot(m_1,m_2),np_l.inv(m_e1))

    alpha = np.dot(m_2,np_l.inv(m_3))

    return alpha
