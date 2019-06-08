"""
Test the LIBSVM loader.

@author: David Diaz Vico
@license: MIT
"""

from . import check_estimator

from skdatasets.repositories.libsvm import fetch


def check(data, shape, test_shape=None, validation_shape=None,
          remaining_shape=None, estimator=True):
    """Check dataset properties."""
    if validation_shape is not None:
        shape = [shape[0] + validation_shape[0], *shape[1:]]
    for a, b in zip(data.data.shape, shape):
        assert a == b
    assert data.target.shape[0] == shape[0]
    if test_shape is not None:
        assert data.data_test.shape == test_shape
        assert data.target_test.shape[0] == test_shape[0]
    else:
        assert data.data_test is None
        assert data.target_test is None
    if validation_shape is not None:
        assert data.inner_cv is not None
    else:
        assert data.inner_cv is None
    assert data.outer_cv is None
    if remaining_shape is not None:
        assert data.data_remaining.shape == remaining_shape
        assert data.target_remaining.shape[0] == remaining_shape[0]
    else:
        assert data.data_remaining is None
        assert data.target_remaining is None
    if estimator:
        check_estimator(data)


def test_fetch_libsvm_australian():
    """Tests LIBSVM australian dataset."""
    data = fetch('binary', 'australian')
    check(data, (690, 14))


def test_fetch_libsvm_liver_disorders():
    """Tests LIBSVM liver-disorders dataset."""
    data = fetch('binary', 'liver-disorders')
    check(data, (145, 5), test_shape=(200, 5))


def test_fetch_libsvm_duke():
    """Tests LIBSVM duke dataset."""
    data = fetch('binary', 'duke')
    check(data, (38, 7129), validation_shape=(4, 7129), estimator=False)


def test_fetch_libsvm_cod_rna():
    """Tests LIBSVM cod-rna dataset."""
    data = fetch('binary', 'cod-rna')
    check(data, (59535, 8), test_shape=(271617, 8), remaining_shape=(157413, 8))


def test_fetch_libsvm_satimage():
    """Tests LIBSVM satimage dataset."""
    data = fetch('multiclass', 'satimage.scale')
    check(data, (3104, 36), test_shape=(2000, 36), validation_shape=(1331, 36))
