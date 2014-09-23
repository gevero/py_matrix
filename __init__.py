# -*- coding: utf-8 -*-
"""
py_matrix - A python implementation of the transfer matrix method
for multilayer structures with arbitrary dielectric tensors. The
code is based on the paper from Masud Mansuripur:

"Analysis of multilayer thin‐film structures containing magneto‐optic
and anisotropic media at oblique incidence using 2×2 matrices,
 J. Appl. Phys. 67, 6466 (1990); http://dx.doi.org/10.1063/1.345121"
"""

# from .core import rt
import py_matrix.core as core
import py_matrix.mat as mat
import py_matrix.utils as utils
import py_matrix.moe as moe
