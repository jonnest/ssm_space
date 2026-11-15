################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
from setuptools import setup

setup(
  name = 'ssm_library',
  packages = ['ssm', 'tools', 'deformetrica'],
  version = '0.0.1',
  license='Proprietary',
  description = 'Statistical Shape Modelling',
  author = 'Jonathan Nestmann',
  keywords = ['Shape Analysis', 'Geometric Statistics'],
  install_requires=[
          'numpy',
          'scipy',
          'seaborn',
          'pandas',
          'matplotlib',
          'pyvista',
          'pymeshlab'
      ],
  classifiers=[
    'Development Status :: 1 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: Other/Proprietary License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
  ],
)