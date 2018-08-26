"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.classification_val_test import load_satimage

from ...base import check_load_dataset


@pytest.mark.slow
def test_satimage():
    """Tests satimage dataset."""
    pass
    n_patterns = (4435, 3104, 1331, 2000)
    n_variables = 36
    array_names = (('data', 'target'), ('data_tr', 'target_tr'),
                   ('data_val', 'target_val'), ('data_test', 'target_test'))
    check_load_dataset(load_satimage, n_patterns, n_variables, array_names,
                       n_targets=None)
