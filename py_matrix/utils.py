import numpy as np
import scipy as sp
z0 = sp.constants.value('characteristic impedance of vacuum')

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

    # extracting values from the matrix
    rt_pp = m_rt_ps[0,0]
    rt_ps = m_rt_ps[0,1]
    rt_sp = m_rt_ps[1,0]
    rt_ss = m_rt_ps[1,1]

    # calculating the values
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

    # extracting values from the matrix
    r_pp = m_r_ps[0,0]
    r_ps = m_r_ps[0,1]
    r_sp = m_r_ps[1,0]
    r_ss = m_r_ps[1,1]

    # calculating the reflectance
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

    # extracting values from the matrix
    t_pp = m_t_ps[0,0]
    t_ps = m_t_ps[0,1]
    t_sp = m_t_ps[1,0]
    t_ss = m_t_ps[1,1]

    # calculating the transmittance
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


def field(m_K,m_E,m_H,e_list_3x3,d_list,x,y,z,pol):
    '''Starting from field amplitudes and wavevectors in each layer, and from
    the multilayer structure (e_list_3x3,d_list), calculates the complex
    electric and magnetic field, and along with that the Poynting vector and
    energy absorption at the location (x,y,z) in the multilayer

    Parameters
    ----------
    'm_K' = wavevectors shape = (n_layers,n_k=4,n_xyz=3)
    'm_E,m_H' = field amplitudes shape = (n_layers,n_k=4,n_xyz=3,n_pol=2)
    'e_list_3x3'= [n_layer+2,3,3] numpy array: it contains n_layers+2 3x3
                  dielectric tensors:
    e_list_3x3[0]= 3x3 incident medium dielectric tensor: must be real,diagonal
                   and isotropic,
    e_list_3x3[n_layers+1] = 3x3 substrate dielectric tensor: must be real,
                             diagonal and isotropic,
    e_list_3x3[n]=3x3 dielectric tensor of the n_th layers: arbitrary
    'd_list'= n_layers+2 numpy array: contains layer thinknesses:
    d_list[0]=d_list[n_layers+1]=0: for the incident medium and substrate
    d_list[n]=d_n n_th layer thickness in nm
    x,y,z = coordinates in nm
    pol = 'TE' or 'TM', polarization state

    Returns
    -------
    'a dictionary'= {'E': v_E,  # electric field vector
                     'H': v_H,  # magnetic field vector
                     'S': v_S,  # Poynting vector
                     'normE': np.linalg.norm(v_E),  # normalized E
                     'normH': np.linalg.norm(v_H),  # normalized H
                     'normS': np.linalg.norm(v_S),  # normalized S
                     'absor':absor}  # absorption at (x,y,z)
'''
    # auxiliary computations
    n_layers = len(e_list_3x3)-2  # recovering number of layers
    v_z = np.cumsum(d_list)[:-1]  # starting coordinate of each layers (except inc)
    n_l = np.count_nonzero(z > v_z)  # current layers

    # output vectors
    v_E = np.zeros(3,dtype=np.complex128)
    v_H = np.zeros(3,dtype=np.complex128)
    v_S = np.zeros(3)
    v_dE = np.zeros(3,dtype=np.complex128)
    v_dH = np.zeros(3,dtype=np.complex128)

    # selecting the polarization
    if pol == 'TE':
        i_p = 0
    else:
        i_p = 1

    # right surface coordinate
    if n_l <= n_layers:
        z_n = v_z[n_l]
    else:
        z_n = v_z[n_l-1]

    # summing the four components of the fields
    for m in range(4):
        v_E = v_E + m_E[n_l,m,:,i_p]*np.exp(1j*(m_K[n_l,m,0]*x +
                                                m_K[n_l,m,1]*y +
                                                m_K[n_l,m,2]*(z-z_n)))
        v_dE = v_dE + 1j*m_K[n_l,m,2]*m_E[n_l,m,:,i_p]*np.exp(1j*(m_K[n_l,m,0]*x +
                                                                  m_K[n_l,m,1]*y +
                                                                  m_K[n_l,m,2]*(z-z_n)))
        v_H = v_H + m_H[n_l,m,:,i_p]*np.exp(1j*(m_K[n_l,m,0]*x +
                                            m_K[n_l,m,1]*y +
                                            m_K[n_l,m,2]*(z-z_n)))
        v_dH = v_dH + 1j*m_K[n_l,m,2]*m_H[n_l,m,:,i_p]*np.exp(1j*(m_K[n_l,m,0]*x +
                                                                  m_K[n_l,m,1]*y +
                                                                  m_K[n_l,m,2]*(z-z_n)))

    v_S = 0.5*np.real(np.cross(v_E,np.conj(v_H/z0)))
    I_abs = (0.5*np.real(np.cross(v_dE,np.conj(v_H/z0)))[2]+
             0.5*np.real(np.cross(v_E,np.conj(v_dH/z0)))[2])

    return {'E': v_E, 'H': v_H,'S': v_S,'normE': np.linalg.norm(v_E), 'normH': np.linalg.norm(v_H),'normS': np.linalg.norm(v_S),'abs':I_abs}
