"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.classification import load_skin_nonskin

from ...base import check_load_dataset


@pytest.mark.slow
def test_skin_nonskin():
    """Tests skin_nonskin dataset."""
    n_patterns = (245057,)
    n_variables = 3
    array_names = (('data', 'target'),)
    check_load_dataset(load_skin_nonskin, n_patterns, n_variables, array_names,
                       n_targets=None)
