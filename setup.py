import sys

from setuptools import find_packages, setup

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []
setup(name='scikit-datasets',
      packages=find_packages(),
      version='0.1.15',
      description='Scikit-learn-compatible datasets',
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-datasets',
      download_url='https://github.com/daviddiazvico/scikit-datasets/archive/v0.1.15.tar.gz',
      keywords=['scikit-learn', 'pandas'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      install_requires=['scikit-learn', 'pandas'],
      extras_require={
        'cran':  ['rdata'],
      },
      setup_requires=pytest_runner,
      tests_require=['pytest-cov'],
      test_suite='tests',
      )
