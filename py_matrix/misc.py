'''misc.py contains untested subroutines: use at your own risk'''


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
    eps_r_1 = (w_p**2)*(1.0/(w**2+gamma_inf**2)-1.0/(w**2+gamma_r**2))
    eps_r_2 = ((w_p**2/w)*(gamma_r/(w**2+gamma_r**2) -
               gamma_inf/(w**2+gamma_inf**2)))

    eps_r = eps_r_1+1j*eps_r_2

    return eps_r
