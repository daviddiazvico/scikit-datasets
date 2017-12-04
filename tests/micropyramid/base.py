"""
Tests.

@author: David Diaz Vico
@license: MIT
"""


def check_load_dataset(load, n_patterns, fdescr):
    """Checks that a dataset is loaded correctly."""

    X = load(return_X_y=True)
    assert len(X) == n_patterns
    bunch = load()
    assert len(bunch.data) == n_patterns
    assert len(bunch.feature_names) == n_patterns
    assert bunch.DESCR == fdescr
