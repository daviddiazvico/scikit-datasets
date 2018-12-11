"""
Test the LIBSVM loader.

@author: David Diaz Vico
@license: MIT
"""

from . import check_estimator

from skdatasets.libsvm import fetch


def check(data, shape, test_shape=None):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if test_shape is not None:
        assert data.data_test.shape == test_shape
        assert data.target_test.shape[0] == test_shape[0]
    if hasattr(data, 'inner_cv') and (data.inner_cv is not None):
        assert len(list(data.inner_cv.split())) == 1
    check_estimator(data)


def test_fetch_libsvm_australian():
    """Tests LIBSVM australian dataset."""
    data = fetch('binary', 'australian')
    check(data, (690*2, 14))


def test_fetch_libsvm_liver_disorders():
    """Tests LIBSVM liver-disorders dataset."""
    data = fetch('binary', 'liver-disorders')
    check(data, (145*2, 5), test_shape=(145, 5))


def test_fetch_libsvm_duke():
    """Tests LIBSVM duke dataset."""
    data = fetch('binary', 'duke')
    check(data, (44*2, 7129))


def test_fetch_libsvm_cod_rna():
    """Tests LIBSVM cod-rna dataset."""
    data = fetch('binary', 'cod-rna')
    check(data, (59535*2, 8))


def test_fetch_libsvm_satimage():
    """Tests LIBSVM satimage dataset."""
    data = fetch('multiclass', 'satimage.scale')
    check(data, (4435*2, 36), test_shape=(4435, 36))
