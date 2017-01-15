
from setuptools import setup

# Utility function to read the README file.
def readme():
    with open('README.md') as f:
        return f.read()

descrip = ("A transfer matrix code for anysotropic multilayers")

data_files = ['README.md','LICENSE.txt','Changes.txt','manual.pdf','examples.ipynb']

setup(
    name = "py_matrix",
    version = '0.0.1',
    author = "Giovanni Pellegrini",
    author_email = "giovi.pelle@gmail.com",
    description = descrip,
    license = "GPL 3.0",
    keywords = "optics, reflection, absorption, photovoltaics, ellipsometry, transfer matrix method",
    url='https://github.com/gevero/py_matrix',
    packages=['py_matrix'],
    long_description=readme(),
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License, version 3.0",
        "Programming Language :: Python :: 3.x"],
)
