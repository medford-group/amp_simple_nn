#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='amp_simple_nn',
      version='0.1',
      description='',
      author='Ben Comer',
      author_email='ben.comer@gatech.edu',
      url='https://github.com/medford-group/amp_simple_nn',
      scripts=['lammps_interface/bin/py_wrapped_packmol.py'],
      packages=find_packages(),
      install_requires=['spglib', 'numpy>=1.16.0','ase'],
     )


