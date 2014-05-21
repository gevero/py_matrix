import numpy as np
import scipy as sp


#---------------------------------------------------
#Computation of rotation and ellipticity
#---------------------------------------------------
def rot_ell(m_rt_sp):
    '''Utility to compute rotation and ellipticity
       starting from reflection and transmission matrix

    Parameters
    ----------
    'm_RTsp' = sp reflection and transmission matrix

    Returns
    -------
    'a dictionary' = {'theta_s':theta_s,
                      'eps_s':eps_s,
                      'theta_p':theta_p,
                      'eps_p':eps_p}
    '''

    #------extracting values from the matrix------
    rt_pp = m_rt_sp[0,0]
    rt_ps = m_rt_sp[0,1]
    rt_sp = m_rt_sp[1,0]
    rt_ss = m_rt_sp[1,1]

    #------calculating the values------
    theta_s = np.real(rt_ps/rt_ss)
    eps_s = np.imag(rt_ps/rt_ss)
    theta_p = np.real(rt_sp/rt_pp)
    eps_p = np.imag(rt_sp/rt_pp)

    out_dict = {'theta_s':theta_s,
                'eps_s':eps_s,
                'theta_p':theta_p,
                'eps_p':eps_p}

    return out_dict


#---------------------------------------------------
#Computation of s and p reflectance
#---------------------------------------------------
def R_sp(m_r_sp):
    '''Utility to compute s and p reflectance

    Parameters
    ----------
    'm_r_sp' = sp reflection matrix

    Returns
    -------
    'a dictionary' = {'R_p':R_p,
                      'R_s':R_s}
    '''

    #------calculating the reflectance------
    R_p = np.abs(m_r_sp[0,0])**2
    R_s = np.abs(m_r_sp[1,1])**2

    out_dict = {'R_p':R_p,
                'R_s':R_s}

    return out_dict


#---------------------------------------------------
#Computation of s and p transmittance
#---------------------------------------------------
def T_sp(m_t_sp,theta_0,n_0,n_s):
    '''Utility to compute s and p reflectance

    Parameters
    ----------
    'm_t_sp' = sp transmission matrix

    Returns
    -------
    'a dictionary' = {'T_p':T_p,
                      'T_s':T_s}
    '''

    #------calculating the transmittance------
    theta_s = sp.arcsin(np.real_if_close(np.sin(theta_0)*n_0/n_s))
    T_p = np.abs(m_t_sp[0,0]**2) * (((n_s*np.conj(np.cos(theta_s))).real) /
                                    (n_0*np.conj(np.cos(theta_0))).real)
    T_s = np.abs(m_t_sp[1,1]**2) * (((n_s*np.cos(theta_s)).real) /
                                    (n_0*np.cos(theta_0)).real)

    out_dict = {'T_p':T_p,
                'T_s':T_s}

    return out_dict
