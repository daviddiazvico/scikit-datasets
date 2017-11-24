"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_cpusmall


def test_cpusmall():
    """Tests cpusmall dataset."""
    n_patterns = (8192, )
    n_variables = 12
    array_names = (('data', 'target'), )
    check_load_dataset(load_cpusmall, n_patterns, n_variables, array_names,
                       n_targets=None)
