py\_matrix
==========

|Binder|

A python implementation of the transfer matrix method for multilayer
structures with arbitrary dielectric tensors. If you find py-matrix
useful for generating results included in publications, please consider
citing the following paper:

Pellegrini G. and Mattei G. (2014) **High-Performance Magneto-Optic
Surface Plasmon Resonance Sensor Design: An Optimization Approach.**
*Plasmonics 1–6*.
`doi:10.1007/s11468-014-9764-6 <http://link.springer.com/article/10.1007/s11468-014-9764-6>`__

Installation
------------

Installing **py\_matrix** should be fairly easy. First of all you need a
python distribution on your system. The simplest thing to do if you are
using **Windows** or **OSX** would be to install
`Anaconda <https://store.continuum.io/cshop/anaconda/>`__, a beautiful,
free, and easy to use python distribution: the relevant installations
instructions are found
`here <http://docs.continuum.io/anaconda/install.html>`__. If you are
using **Linux** you probably already know how to install python on your
system, nevertheless my advice is to also install `The IPython
Notebook <http://ipython.org/notebook.html>`__, which is already bundled
in Anaconda.

The second step would be installing **py\_matrix** itself. If you are
familiar with `git <http://git-scm.com/>`__ and
`github <https://github.com/>`__ you can simply clone the repository,
otherwise just
`download <https://github.com/gevero/py-matrix/archive/master.zip>`__
the zipped version of the repo and unpack it wherever you like.

Usage
-----

The best thing to do for the moment is to start from the **.ipynb**
files in the
`examples <https://github.com/gevero/py-matrix/tree/master/examples>`__
folder. You can load them in your local IPython Notebook instance: they
should give you a fair idea about how to proceed for your calculations.
Each function in the code features a detailed documentation easily
accessible with the ``Shift-Tab`` tool-tip shortcut from the IPython
Notebook interface.

.. |Binder| image:: http://mybinder.org/badge.svg
   :target: http://mybinder.org:/repo/gevero/py_matrix
