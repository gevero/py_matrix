# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

#  # Gold Surface Plasmon Resonance (SPR) Example
#  The notebook is structured as follows:
#  -  Setup of useful settings and import of necessary libraries
#  -  Inputs for the simulation
#  -  Computation
#  - Plot

# <markdowncell>

# ## Settings and libraries

# <codecell>

# inline plot magic
%pylab inline

import numpy as np # numpy
import sys                  # sys to add py_matrix to the path

# adding py_matrix parent folder to python path
sys.path.append('/path/to/py_matrix/parent/folder')
import py_matrix as pm # importing py_matrix

# useful parameters
f_size=20;

# <markdowncell>

# ## Inputs
# - Loading optical constants and building the optical constant database
# - Setting the inputs such as layer compositionm thickness, incident angles, etc...

# <codecell>

# building the optical constant database, point the folder below to the "materials" py_matrix folder
eps_db_out=pm.mat.generate_eps_db('path/to/optical_constants/folder/',ext='*.edb')
eps_files,eps_names,eps_db=eps_db_out['eps_files'],eps_db_out['eps_names'],eps_db_out['eps_db']

# <codecell>

stack=['e_bk7','e_au','e_vacuum']; # materials composing the stack, as taken from eps_db
d_list=[0.0,55.0,0.0]; # multilayer thicknesses: incident medium and substrate have zero thickness
wl_0=633; # incident wavelenght in nm
# polar angle in radians
theta_min=40*np.pi/1.8e2;
theta_max=50*np.pi/1.8e2;
theta_step=500;
v_theta=np.linspace(theta_min,theta_max,theta_step)
# azimuthal angle radians
phi_0=0.0

# <markdowncell>

# ## Computation
# - Retrieval of optical constants at $\lambda$=633 nm from the optical constant database
# - Filling of the dielectric tensor at $\lambda$=633 nm
# - Initialization of the reflectance output vector
# - Polar angle loop

# <codecell>

# optical constant tensor
m_eps=np.zeros((len(stack),3,3),dtype=np.complex128);
e_list=pm.mat.db_to_eps(wl_0,eps_db,stack) # retrieving optical constants at wl_0 from the database
m_eps[:,0,0]=e_list # filling dielectric tensor diagonal
m_eps[:,1,1]=e_list
m_eps[:,2,2]=e_list

# initializing reflectance output vector
v_r_p=np.zeros_like(v_theta)

# angle loop
for i_t,t in enumerate(v_theta):
        
    #------Computing------
    m_r_ps=pm.core.rt(wl_0,t,phi_0,m_eps,d_list)['m_r_ps'] # reflection matrix
    v_r_p[i_t]=pm.utils.R_ps_rl(m_r_ps)['R_p'] # getting p-polarized reflectance

# <markdowncell>

# ## Plot of the reflectance spectrum at $\lambda$ = 633 nm

# <codecell>

plt.figure(1,figsize=(15,10))
plt.plot(v_theta*1.8e2/np.pi,v_r_p,'k',linewidth=2.0)

# labels
plt.xlabel(r'$\Theta^{\circ}$',fontsize=f_size+10)
plt.ylabel('R',fontsize=f_size+10)

# ticks
plt.xticks(fontsize=f_size)
plt.yticks(fontsize=f_size)

# grids
plt.grid()

#legends
plt.legend(['55 nm Au film reflectance at 633 nm'],loc='upper right',fontsize=f_size,frameon=False);

