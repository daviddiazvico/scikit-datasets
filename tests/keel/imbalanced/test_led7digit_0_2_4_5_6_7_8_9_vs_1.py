"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_led7digit_0_2_4_5_6_7_8_9_vs_1


def test_load_led7digit_0_2_4_5_6_7_8_9_vs_1():
    """Tests led7digit-0-2-4-5-6-7-8-9_vs_1 dataset."""
    n_patterns = (443, )
    n_variables = 7
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_led7digit_0_2_4_5_6_7_8_9_vs_1, n_patterns,
                       n_variables, array_names, n_targets=None,
                       n_folds=n_folds)
