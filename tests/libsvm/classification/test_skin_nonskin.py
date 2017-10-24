"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification import load_skin_nonskin


def test_skin_nonskin():
    """Tests skin_nonskin dataset."""
    n_patterns = (245057, )
    n_variables = 3
    array_names = (('data', 'target'), )
    check_load_dataset(load_skin_nonskin, n_patterns, n_variables, array_names,
                       n_targets=None)
