#! /usr/bin/env python

"""
@author: David Diaz Vico
@license: MIT
"""

from setuptools import find_packages, setup

setup(
    name='scikit-datasets',
    packages=find_packages(),
    version='0.2.0',
    description='Scikit-learn-compatible datasets',
    author='David Diaz Vico',
    author_email='david.diaz.vico@outlook.com',
    url='https://github.com/daviddiazvico/scikit-datasets',
    download_url='https://github.com/daviddiazvico/scikit-datasets/archive/v0.2.0.tar.gz',
    keywords=['scikit-learn'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    extras_require={
        'cran': ['rdata'],
        'forex': ['forex_python>=1.6'],
        'keel': ['pandas'],
        'keras': ['keras'],
        'utils.estimator': ['jsonpickle'],
        'utils.experiments': ['sacred'],
        'utils.scores': ['statsmodels'],
    },
    setup_requires=['pytest-runner'],
    tests_require=[
        'coverage',
        'forex_python>=1.6',
        'jsonpickle',
        'keras',
        'pandas',
        'pymongo',
        'pytest',
        'pytest-cov',
        'rdata',
        'sacred',
        'statsmodels',
        'tensorflow',
    ],
    test_suite='skdatasets.tests',
)
