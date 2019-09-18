#! /usr/bin/env python

"""
@author: David Diaz Vico
@license: MIT
"""

from setuptools import find_packages, setup

setup(name='scikit-datasets',
      packages=find_packages(),
      version='0.1.32',
      description='Scikit-learn-compatible datasets',
#      long_description=open('README.md', 'r').read(),
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-datasets',
      download_url='https://github.com/daviddiazvico/scikit-datasets/archive/v0.1.32.tar.gz',
      keywords=['scikit-learn'],
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      extras_require={'cran':  ['rdata'],
                      'forex': ['forex_python'],
                      'keel': ['pandas'],
                      'keras': ['keras'],
                      'utils.estimator': ['jsonpickle==0.9.6'],
                      'utils.experiments': ['sacred']},
      setup_requires=['pytest-runner'],
      tests_require=['coverage', 'forex_python', 'jsonpickle==0.9.6', 'keras',
                     'pandas', 'pymongo', 'pytest', 'pytest-cov', 'rdata',
                     'sacred', 'tensorflow'],
      test_suite='tests')
