'''mat.py contains subroutines to create, tabulate and manipulate optical
constants'''

import numpy as np
import numpy.linalg as np_l
import scipy.constants as sp_c
import glob
import os


def import_eps(eps_file):
    '''Utility to read in optical constants, i.e.
    dielectric functions

    Parameters
    ----------
    'eps_file' = optical constant filename with
    full path

    Returns
    -------
    'e_raw[:,0]' = wavelength
    'e_raw[:,1]' = real part of the dielectric function
    'e_raw[:,2]' = imaginari part of the dielectric function
    '''

    e_raw = np.loadtxt(eps_file,skiprows=1)
    return e_raw[:,0],e_raw[:,1],e_raw[:,2]


def generate_eps_db(path_to_eps,ext="*.edb"):
    '''Utility to build a database of optical constants

    Parameters
    ----------
    'path_to_eps' = path to the folder containing the
    files of the optical constants
    'ext' = extension of the files of the optical
    constants. "*.edb" is the default extension for
    the optical constant files


    Returns
    -------
    'a dictionary' = {'eps_files':eps_files, = fullpath optical constant file
                                               names
                      'eps_names':eps_names, = optical constants formal names
                      'eps_db':eps_db}       = optical constant database
                                               dictionary
    '''

    eps_files = sorted(glob.glob(path_to_eps+ext))
    eps_names = [os.path.splitext(os.path.basename(eps))[0] for
                 eps in eps_files]

    # Dielectric function dictionary database
    eps_db = {}
    for eps_name,eps_file in zip(eps_names,eps_files):
        eps_db[eps_name] = import_eps(eps_file)

    eps_dict = {'eps_files':eps_files,
                'eps_names':eps_names,
                'eps_db':eps_db}

    return eps_dict


def db_to_eps(wl,eps_db,eps_list):
    '''Utility to do the following: it takes the optical constant database
    created with generate_eps_db end a list of optical constant names (taken
    from eps_names) and spits out a numpy.array of complex128 values, i.e. a
    vector containing the multilayer optical constants at the wavelength wl. It
    may come in handy when your multilayer is made of materials whose optical
    constants are already tabulated

    Parameters
    ----------
    'wl' = interpolation wavelength
    'eps_db' = optical constant database as from generate_eps_db
    'eps_list' = a list of optical constant filenames taken from the output of
    generate_eps_db

    Returns
    -------
    'e_list' = list of multilayer optical constants'''

    # checking wavelength
    for eps in eps_list:
        if (wl < eps_db[eps][0].min()):
            raise ValueError('interpolation wavelength too small: '
                             + eps + '(' + str(wl) + '/' +
                             str(eps_db[eps][0].min()) + ')')
        if (wl > eps_db[eps][0].max()):
            raise ValueError('interpolation wavelength too large: '
                             + eps + '(' + str(wl) + '/' +
                             str(eps_db[eps][0].max()) + ')')

    # dielectric constant array
    e_list = np.array([np.interp(wl,eps_db[eps][0],eps_db[eps][1]) +
                      1j*np.interp(wl,eps_db[eps][0],eps_db[eps][2])
                      for eps in eps_list])

    return e_list


def eps_drude(wl,eps_inf,w_p,gamma):
    '''computation of the drude dielectric function

    Parameters
    ----------
    'wl' = computation wavelength (nm)
    'eps_inf' = high frequency constant, for frequency much higher than the
    bulk plasma ones
    'w_p' = bulk plasma frequency (eV)
    'gamma' = damping constant (eV)

    Returns
    -------
    'eps' = drude optical constant at the wavelength wl'''

    # computation

    # nm to eV
    w = 1240.0/wl
    eps = eps_inf-(w_p**2)/(w**2+1j*w*gamma)

    return eps


def eps_corr_drude(wl,w_p,gamma_inf,vf,r):
    '''computation of the drude dielectric function correction
    for spherical particles of finite size as in:
    Kreibig Vollmer "Optical properties of metal clusters"

    Parameters
    ----------
    'wl' = computation wavelength (nm)
    'w_p' = bulk plasma frequency (eV)
    'gamma_inf' = bulk damping constant (eV)
    'vf' = Fermi velocity (m/s)
    'r' = particle radius (nm)

    Returns
    -------
    'eps_r' = drude optical constant size correction'''

    # computation
    h_bar = 6.58211928e-16  # h_bar eV
    gamma_r = gamma_inf+h_bar*vf/(r*1e-9)
    w = 1240.0/wl  # nm to eV

    eps_r = (w_p**2)/(w**2 + 1j*w*gamma_inf) - (w_p**2)/(w**2 + 1j*w*gamma_r)

    return eps_r


def eps_xy_drude(wl,w_p,gamma,B,f_m=0.0):
    '''computation of the off diagonal optical constant for a noble metal
    according to a free electron model

    Parameters
    ----------
    'wl' = computation wavelength (nm)
    'w_p' = bulk plasma frequency (eV)
    'gamma' = damping constant (eV)
    'B' = magnetic field (Tesla)
    'f_m' = Verdet constant

    Returns
    -------
    'eps_xy' = off diagonal drude optical constant at the wavelength wl'''

    # computation
    h_bar = 6.58211928e-16  # h_bar eV
    w = 1240.0/wl  # nm to eV
    m_e = sp_c.m_e  # electron mass
    e_el = sp_c.e  # elementary charge
    w_c = -h_bar*(e_el*B)/m_e  # cyclotron frequency (rad/s to eV)
    eps_xy = 1j*((w_p**2)*w_c)/(w*(w+1j*gamma)**2)

    return eps_xy


def m_eff_MG(m_L,m_D,V,m_e1,m_e2,w_l,f):
    '''calculates the MG effective dielectric tensor for the ellipsoid layer

    Parameters
    ----------
    'm_L,m_D' = static and dynamic geometrical tensor of the ellipsoid
    'V'=volume of the ellipsoind
    'm_e1,m_e2' = diagonal tensor of the embedding matrix,
                  Magneto Optical tensor of the ellipsoid
    'w_l' = incident wavelength in nm
    'f' = filling factor

    Returns
    -------
    'eff_MG' = effective dielectric tensor of the mixed material
               containing the ellipsoids'''

    m_I = np.identity(3)
    k = 2.0*np.pi*np.sqrt(m_e1[2,2])/w_l

    m_1 = (1-f)*(m_L - ((k**2)*V*m_D/(4.0*np.pi)) -
                 1j*((k**3)*V*m_I/(6.0*np.pi)))
    m_2 = m_e2-m_e1
    m_3 = m_I+np.dot(np.dot(m_1,m_2),np_l.inv(m_e1))

    eff_MG = m_e1+np.dot(f*m_2,np_l.inv(m_3))

    return eff_MG
