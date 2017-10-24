"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.uci.classification import load_nursery


def test_nursery():
    """Tests nursery dataset."""
    n_patterns = (12960, )
    n_variables = 27
    array_names = (('data', 'target'), )
    check_load_dataset(load_nursery, n_patterns, n_variables, array_names,
                       n_targets=None)
