import numpy as np
import scipy as sp


def rot_ell(m_rt_ps):
    '''Utility to compute rotation and ellipticity
       starting from reflection and transmission matrix

    Parameters
    ----------
    'm_RTsp' = sp reflection and transmission matrix

    Returns
    -------
    'a dictionary' = {'theta_p':theta_p,
                      'eps_p':eps_p,
                      'theta_s':theta_s,
                      'eps_s':eps_s}
    '''

    #------extracting values from the matrix------
    rt_pp = m_rt_ps[0,0]
    rt_ps = m_rt_ps[0,1]
    rt_sp = m_rt_ps[1,0]
    rt_ss = m_rt_ps[1,1]

    #------calculating the values------
    theta_p = np.real(rt_sp/rt_pp)
    eps_p = np.imag(rt_sp/rt_pp)
    theta_s = np.real(rt_ps/rt_ss)
    eps_s = np.imag(rt_ps/rt_ss)

    out_dict = {'theta_p':theta_p,
                'eps_p':eps_p,
                'theta_s':theta_s,
                'eps_s':eps_s}

    return out_dict


def R_ps_rl(m_r_ps):
    '''Utility to compute reflectance for p,s and right,left
    circular polarization

    Parameters
    ----------
    'm_r_ps' = ps reflection matrix

    Returns
    -------
    'a dictionary' = {'R_p':R_p, #reflectances
                      'R_s':R_s,
                      'R_l':R_l,
                      'R_r':R_r}
    '''

    #------extracting values from the matrix------
    r_pp = m_r_ps[0,0]
    r_ps = m_r_ps[0,1]
    r_sp = m_r_ps[1,0]
    r_ss = m_r_ps[1,1]

    #------calculating the reflectance------
    R_p = np.abs(r_pp)**2 + np.abs(r_sp)**2
    R_s = np.abs(r_ss)**2 + np.abs(r_ps)**2
    R_r = 0.5*(np.abs(r_pp)**2 + np.abs(r_ss)**2 +
               np.abs(r_sp)**2 + np.abs(r_ps)**2 +
               np.real(1j*r_pp*np.conj(r_ps)) +
               np.real(1j*r_sp*np.conj(r_ss)))
    R_l = 0.5*(np.abs(r_pp)**2 + np.abs(r_ss)**2 +
               np.abs(r_sp)**2 + np.abs(r_ps)**2 -
               np.real(1j*r_pp*np.conj(r_ps)) -
               np.real(1j*r_sp*np.conj(r_ss)))

    out_dict = {'R_p':R_p,
                'R_s':R_s,
                'R_r':R_r,
                'R_l':R_l}

    return out_dict


def T_ps_rl(m_t_ps,theta_0,n_0,n_s):
    '''Utility to compute transmittance for p,s and right.left
    circular polarization

    Parameters
    ----------
    'm_t_ps' = ps transmission matrix

    Returns
    -------
    'a dictionary' = {'T_p':T_p, #transmittances
                      'T_s':T_s,
                      'T_l':T_l,
                      'T_r':T_r
                      'A_p':-np.log10(T_p), #absorbances
                      'A_s':-np.log10(T_s),
                      'A_r':-np.log10(T_r),
                      'A_l':-np.log10(T_l)}
    '''

    #------extracting values from the matrix------
    t_pp = m_t_ps[0,0]
    t_ps = m_t_ps[0,1]
    t_sp = m_t_ps[1,0]
    t_ss = m_t_ps[1,1]

    #------calculating the transmittance------
    theta_s = sp.arcsin(np.real_if_close(np.sin(theta_0)*n_0/n_s))
    norm = (np.real(n_s*np.conj(np.cos(theta_s))) /
            np.real(n_0*np.conj(np.cos(theta_0))))
    T_p = norm*(np.abs(t_pp)**2 + np.abs(t_sp)**2)
    T_s = norm*(np.abs(t_ss)**2 + np.abs(t_ps)**2)
    T_r = 0.5*norm*(np.abs(t_pp)**2 + np.abs(t_ss)**2 +
                    np.abs(t_sp)**2 + np.abs(t_ps)**2 +
                    2.0*np.real(1j*t_pp*np.conj(t_ps)) +
                    2.0*np.real(1j*t_sp*np.conj(t_ss)))
    T_l = 0.5*norm*(np.abs(t_pp)**2 + np.abs(t_ss)**2 +
                    np.abs(t_sp)**2 + np.abs(t_ps)**2 -
                    2.0*np.real(1j*t_pp*np.conj(t_ps)) -
                    2.0*np.real(1j*t_sp*np.conj(t_ss)))

    out_dict = {'T_p':T_p,
                'T_s':T_s,
                'T_l':T_l,
                'T_r':T_r,
                'A_p':-np.log10(T_p),
                'A_s':-np.log10(T_s),
                'A_r':-np.log10(T_r),
                'A_l':-np.log10(T_l)}

    return out_dict
