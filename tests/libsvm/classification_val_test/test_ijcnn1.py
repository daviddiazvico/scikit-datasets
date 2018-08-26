"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.classification_val_test import load_ijcnn1

from ...base import check_load_dataset


@pytest.mark.slow
def test_ijcnn1():
    """Tests ijcnn1 dataset."""
    n_patterns = (49990, 35000, 14990, 91701)
    n_variables = 22
    array_names = (('data', 'target'), ('data_tr', 'target_tr'),
                   ('data_val', 'target_val'), ('data_test', 'target_test'))
    check_load_dataset(load_ijcnn1, n_patterns, n_variables, array_names,
                       n_targets=None)
